from PIL import Image
import serial
import piexif
import exifread
import time
import re

def parse_nmea_coordinate(coord_str, direction):
    """
    Convert NMEA coordinate format (DDMM.MMMM) to decimal degrees
    """
    if not coord_str or coord_str == '':
        return None
    
    # NMEA format: DDMM.MMMM for latitude, DDDMM.MMMM for longitude
    coord_float = float(coord_str)
    
    # Extract degrees and minutes
    if len(coord_str.split('.')[0]) == 4:  # Latitude (DDMM)
        degrees = int(coord_float // 100)
        minutes = coord_float % 100
    else:  # Longitude (DDDMM)
        degrees = int(coord_float // 100)
        minutes = coord_float % 100
    
    decimal_degrees = degrees + minutes / 60.0
    
    # Apply direction (negative for South/West)
    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees
    
    return decimal_degrees

def read_gps():
    """
    Read GPS coordinates from serial GPS device
    Returns tuple (latitude, longitude) or None if no fix
    """
    try:
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        print("Aguardando sinal GPS...")
        
        attempts = 0
        max_attempts = 50  # Try for about 50 seconds
        
        while attempts < max_attempts:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            
            if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                parts = line.split(',')
                
                # Check if we have a valid fix (quality indicator > 0)
                if len(parts) >= 7 and parts[6] and int(parts[6]) > 0:
                    try:
                        # Extract coordinates
                        lat_str = parts[2]
                        lat_dir = parts[3]
                        lon_str = parts[4]
                        lon_dir = parts[5]
                        
                        if lat_str and lon_str and lat_dir and lon_dir:
                            lat = parse_nmea_coordinate(lat_str, lat_dir)
                            lon = parse_nmea_coordinate(lon_str, lon_dir)
                            
                            if lat is not None and lon is not None:
                                ser.close()
                                return lat, lon
                        
                    except (ValueError, IndexError) as e:
                        print(f"Erro ao processar coordenadas: {e}")
            
            attempts += 1
            time.sleep(1)
        
        ser.close()
        print("Timeout: Não foi possível obter coordenadas GPS válidas")
        return None
        
    except serial.SerialException as e:
        print(f"Erro ao abrir porta serial: {e}")
        return None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None

def read_existing_gps_from_image(image_path):
    """
    Read existing GPS data from image EXIF
    Returns tuple (latitude, longitude) or None
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # Check if GPS data exists
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
        print(f"Erro ao ler GPS da imagem: {e}")
    
    return None

def to_dms(degree):
    """
    Convert decimal degrees to degrees, minutes, seconds format for EXIF
    """
    degree = abs(degree)
    d = int(degree)
    m = int((degree - d) * 60)
    s = ((degree - d) * 60 - m) * 60
    
    # Convert to rational numbers (numerator, denominator)
    return ((d, 1), (m, 1), (int(s * 1000), 1000))

def add_gps_to_image(image_path, output_path, lat, lon):
    """
    Add GPS coordinates to image EXIF data
    """
    try:
        # Create GPS IFD
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: 'S' if lat < 0 else 'N',
            piexif.GPSIFD.GPSLatitude: to_dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: 'W' if lon < 0 else 'E',
            piexif.GPSIFD.GPSLongitude: to_dms(lon),
            piexif.GPSIFD.GPSMapDatum: "WGS-84"
        }
        
        # Try to preserve existing EXIF data
        try:
            exif_dict = piexif.load(image_path)
            exif_dict["GPS"] = gps_ifd
        except:
            # Create new EXIF dict if none exists
            exif_dict = {"GPS": gps_ifd}
        
        exif_bytes = piexif.dump(exif_dict)
        
        # Open and save image with GPS data
        img = Image.open(image_path)
        img.save(output_path, exif=exif_bytes, quality=95)
        
        print(f"GPS adicionado com sucesso!")
        print(f"Latitude: {lat:.6f}")
        print(f"Longitude: {lon:.6f}")
        print(f"Imagem salva como: {output_path}")
        
    except Exception as e:
        print(f"Erro ao adicionar GPS à imagem: {e}")

def main():
    """
    Main function - demonstrates usage
    """
    image_path = 'minha_foto.jpg'
    output_path = 'foto_com_gps.jpg'
    
    print("=== GPS para EXIF - Processador de Imagens ===\n")
    
    # First, try to read existing GPS from image
    print("1. Verificando GPS existente na imagem...")
    existing_location = read_existing_gps_from_image(image_path)
    
    if existing_location:
        print(f"GPS encontrado na imagem:")
        print(f"Latitude: {existing_location[0]:.6f}")
        print(f"Longitude: {existing_location[1]:.6f}\n")
        
        use_existing = input("Usar coordenadas existentes? (s/n): ").lower()
        if use_existing == 's':
            lat, lon = existing_location
        else:
            # Read from GPS device
            print("2. Lendo coordenadas do GPS...")
            gps_location = read_gps()
            if gps_location:
                lat, lon = gps_location
            else:
                print("Falha ao obter coordenadas GPS.")
                return
    else:
        print("Nenhum GPS encontrado na imagem.")
        print("2. Lendo coordenadas do GPS...")
        gps_location = read_gps()
        if gps_location:
            lat, lon = gps_location
        else:
            print("Falha ao obter coordenadas GPS.")
            return
    
    # Add GPS to image
    print("3. Adicionando GPS à imagem...")
    add_gps_to_image(image_path, output_path, lat, lon)

if __name__ == "__main__":
    main()