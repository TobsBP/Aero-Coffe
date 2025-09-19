import os
import glob
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import piexif
import exifread
from inference import get_model
import supervision as sv
from datetime import datetime
import json

class GPSDetectionHeatmap:
    def __init__(self, model_id="taylor-swift-records/3"):
        """
        Initialize the heatmap generator with a detection model
        """
        self.model = get_model(model_id=model_id)
        self.detection_data = []
        
    def read_gps_from_image(self, image_path):
        """
        Extract GPS coordinates from image EXIF data
        Returns tuple (latitude, longitude) or None
        """
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                
                if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                    lat_ref = str(tags.get('GPS GPSLatitudeRef', ''))
                    lon_ref = str(tags.get('GPS GPSLongitudeRef', ''))
                    
                    lat_vals = tags['GPS GPSLatitude'].values
                    lon_vals = tags['GPS GPSLongitude'].values
                    
                    # Convert from DMS to decimal
                    lat = float(lat_vals[0]) + float(lat_vals[1])/60 + float(lat_vals[2])/3600
                    lon = float(lon_vals[0]) + float(lon_vals[1])/60 + float(lon_vals[2])/3600
                    
                    if lat_ref == 'S':
                        lat = -lat
                    if lon_ref == 'W':
                        lon = -lon
                    
                    return lat, lon
        except Exception as e:
            print(f"Erro ao ler GPS de {image_path}: {e}")
        
        return None
    
    def process_single_image(self, image_path):
        """
        Process a single image: extract GPS, run detection, return results
        """
        try:
            # Get GPS coordinates
            gps_coords = self.read_gps_from_image(image_path)
            if not gps_coords:
                print(f"Sem GPS em {os.path.basename(image_path)}")
                return None
            
            lat, lon = gps_coords
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erro ao carregar {image_path}")
                return None
            
            # Run inference
            results = self.model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            
            # Count detections by class
            detection_counts = {}
            if len(detections) > 0:
                for class_id in detections.class_id:
                    class_name = results.predictions[0].class_name if hasattr(results.predictions[0], 'class_name') else f"class_{class_id}"
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            result = {
                'image_path': image_path,
                'latitude': lat,
                'longitude': lon,
                'total_detections': len(detections),
                'detection_counts': detection_counts,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ {os.path.basename(image_path)}: {len(detections)} detecções em ({lat:.6f}, {lon:.6f})")
            return result
            
        except Exception as e:
            print(f"Erro processando {image_path}: {e}")
            return None
    
    def process_image_folder(self, folder_path, extensions=['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']):
        """
        Process all images in a folder
        """
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not image_files:
            print(f"Nenhuma imagem encontrada em {folder_path}")
            return
        
        print(f"Processando {len(image_files)} imagens...")
        
        for image_path in image_files:
            result = self.process_single_image(image_path)
            if result:
                self.detection_data.append(result)
        
        print(f"Processamento concluído: {len(self.detection_data)} imagens com GPS")
    
    def create_basic_heatmap(self, output_path="detection_heatmap.html", zoom_start=10):
        """
        Create a basic heatmap showing detection density
        """
        if not self.detection_data:
            print("Nenhum dado para criar heatmap")
            return
        
        # Calculate center point
        lats = [item['latitude'] for item in self.detection_data]
        lons = [item['longitude'] for item in self.detection_data]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        # Prepare heatmap data (lat, lon, intensity)
        heat_data = []
        for item in self.detection_data:
            # Use total detections as intensity
            intensity = max(1, item['total_detections'])
            heat_data.append([item['latitude'], item['longitude'], intensity])
        
        # Add heatmap layer
        HeatMap(heat_data, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m)
        
        # Add markers for each detection point
        for item in self.detection_data:
            popup_text = f"""
            <b>Imagem:</b> {os.path.basename(item['image_path'])}<br>
            <b>Detecções:</b> {item['total_detections']}<br>
            <b>Classes:</b> {', '.join([f'{k}: {v}' for k, v in item['detection_counts'].items()])}
            """
            
            folium.CircleMarker(
                location=[item['latitude'], item['longitude']],
                radius=5 + item['total_detections'],
                popup=folium.Popup(popup_text, max_width=300),
                color='red',
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)
        
        # Save map
        m.save(output_path)
        print(f"Heatmap salvo em: {output_path}")
        return m
    
    def create_class_specific_heatmap(self, target_class=None, output_path="class_heatmap.html"):
        """
        Create heatmap for a specific detection class
        """
        if not self.detection_data:
            print("Nenhum dado para criar heatmap")
            return
        
        # Calculate center point
        lats = [item['latitude'] for item in self.detection_data]
        lons = [item['longitude'] for item in self.detection_data]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Filter data for specific class
        heat_data = []
        for item in self.detection_data:
            if target_class:
                intensity = item['detection_counts'].get(target_class, 0)
            else:
                intensity = item['total_detections']
            
            if intensity > 0:
                heat_data.append([item['latitude'], item['longitude'], intensity])
        
        if heat_data:
            # Add heatmap layer
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            # Add markers
            for item in self.detection_data:
                intensity = item['detection_counts'].get(target_class, 0) if target_class else item['total_detections']
                if intensity > 0:
                    popup_text = f"""
                    <b>Imagem:</b> {os.path.basename(item['image_path'])}<br>
                    <b>{target_class if target_class else 'Total'}:</b> {intensity}<br>
                    <b>Todas as classes:</b> {', '.join([f'{k}: {v}' for k, v in item['detection_counts'].items()])}
                    """
                    
                    folium.CircleMarker(
                        location=[item['latitude'], item['longitude']],
                        radius=3 + intensity * 2,
                        popup=folium.Popup(popup_text, max_width=300),
                        color='blue' if target_class else 'red',
                        fillColor='blue' if target_class else 'red',
                        fillOpacity=0.6
                    ).add_to(m)
        
        # Save map
        m.save(output_path)
        print(f"Heatmap específico salvo em: {output_path}")
        return m
    
    def export_data(self, output_path="detection_data.json"):
        """
        Export detection data to JSON
        """
        with open(output_path, 'w') as f:
            json.dump(self.detection_data, f, indent=2)
        print(f"Dados exportados para: {output_path}")
    
    def get_detection_summary(self):
        """
        Print summary of detection data
        """
        if not self.detection_data:
            print("Nenhum dado disponível")
            return
        
        total_images = len(self.detection_data)
        total_detections = sum(item['total_detections'] for item in self.detection_data)
        
        # Count all classes
        all_classes = {}
        for item in self.detection_data:
            for class_name, count in item['detection_counts'].items():
                all_classes[class_name] = all_classes.get(class_name, 0) + count
        
        print(f"\n=== RESUMO DAS DETECÇÕES ===")
        print(f"Total de imagens processadas: {total_images}")
        print(f"Total de detecções: {total_detections}")
        print(f"Média de detecções por imagem: {total_detections/total_images:.2f}")
        print(f"\nClasses detectadas:")
        for class_name, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")

def main():
    """
    Example usage
    """
    # Initialize heatmap generator
    heatmap_gen = GPSDetectionHeatmap(model_id="taylor-swift-records/3") # images/
    
    # Process images from folder
    image_folder = "photos_with_gps"  # Change to your folder path
    heatmap_gen.process_image_folder(image_folder)
    
    # Show summary
    heatmap_gen.get_detection_summary()
    
    # Create heatmaps
    if heatmap_gen.detection_data:
        # Basic heatmap
        heatmap_gen.create_basic_heatmap("detection_heatmap.html")
        
        # Class-specific heatmap (example for first detected class)
        all_classes = set()
        for item in heatmap_gen.detection_data:
            all_classes.update(item['detection_counts'].keys())
        
        if all_classes:
            first_class = list(all_classes)[0]
            heatmap_gen.create_class_specific_heatmap(first_class, f"{first_class}_heatmap.html")
        
        # Export data
        heatmap_gen.export_data("detection_results.json")
    
    print("\nProcessamento concluído!")

if __name__ == "__main__":
    main()