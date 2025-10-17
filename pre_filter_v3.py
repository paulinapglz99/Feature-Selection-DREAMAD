# pre_filter_v3.py
import pandas as pd
import sys
import time
import numpy as np # Importamos numpy para usarlo en la selección de tipos

def get_headers_efficiently(filepath):
    """
    Lee solo la primera línea de un archivo de texto y la divide por comas.
    Es más eficiente para archivos con encabezados masivos.
    """
    print("Leyendo encabezado con método optimizado...")
    start_time = time.time()
    with open(filepath, 'r') as f:
        header_line = f.readline()
    
    headers = header_line.strip().split(',')
    end_time = time.time()
    
    print(f"-> Lectura de encabezado completada en {end_time - start_time:.2f} segundos.")
    return headers

def aggressive_pre_filtering(filepath, chunk_size=5000, completeness_threshold=0.8, variance_threshold=0.01):
    """
    Lee un CSV masivo en chunks de columnas, aplica filtros de completitud y varianza,
    y devuelve una lista de columnas que pasan el filtro.
    """
    print("Iniciando pre-filtrado agresivo...")
    
    headers = get_headers_efficiently(filepath)
    num_total_cols = len(headers)
    print(f"Se encontraron {num_total_cols} columnas en total.")

    surviving_columns = []
    chunks_start_time = time.time()
    
    # Procesamos las columnas en chunks
    for i in range(0, num_total_cols, chunk_size):
        chunk_start_time = time.time()
        
        chunk_cols = headers[i:i + chunk_size]
        
        if not chunk_cols:
            continue
            
        print(f"\nProcesando chunk de columnas {i+1}-{min(i+chunk_size, num_total_cols)}...")
        
        df_chunk = pd.read_csv(filepath, usecols=chunk_cols, low_memory=False)
        
        # --- CAMBIO CLAVE: Seleccionar solo columnas numéricas para los cálculos ---
        # Esto evita el error TypeError al intentar calcular la varianza en columnas de texto.
        numeric_chunk = df_chunk.select_dtypes(include=np.number)

        if numeric_chunk.empty:
            print("  > No se encontraron columnas numéricas en este chunk. Saltando...")
            continue
        
        # 1. Filtro de Completitud (sobre el chunk numérico)
        completeness = (numeric_chunk != 0).sum() / len(numeric_chunk)
        cols_passing_completeness = completeness[completeness >= completeness_threshold].index
        
        # 2. Filtro de Varianza (sobre las que pasaron completitud)
        if not cols_passing_completeness.empty:
            variances = numeric_chunk[cols_passing_completeness].var()
            cols_passing_variance = variances[variances > variance_threshold].index.tolist()
            
            surviving_columns.extend(cols_passing_variance)
            chunk_end_time = time.time()
            print(f"  > Sobrevivieron {len(cols_passing_variance)} columnas en este chunk.")
            print(f"  > Tiempo del chunk: {chunk_end_time - chunk_start_time:.2f} segundos.")
            print(f"  > Total de columnas sobrevivientes hasta ahora: {len(surviving_columns)}")
        else:
            chunk_end_time = time.time()
            print("  > Ninguna columna sobrevivió en este chunk.")
            print(f"  > Tiempo del chunk: {chunk_end_time - chunk_start_time:.2f} segundos.")
            
    chunks_end_time = time.time()
    print(f"\n--- Procesamiento de todos los chunks finalizado en {chunks_end_time - chunks_start_time:.2f} segundos ---")

    return surviving_columns

def create_filtered_dataset(original_filepath, columns_to_keep, output_filepath):
    """
    Crea un nuevo archivo CSV conteniendo solo las columnas seleccionadas.
    """
    print(f"\nSe encontraron un total de {len(columns_to_keep)} columnas sobrevivientes para guardar.")
    if not columns_to_keep:
        print("Advertencia: No hay columnas sobrevivientes. No se creará ningún archivo de salida.")
        return
        
    print(f"Creando el nuevo dataset filtrado con {len(columns_to_keep)} columnas. Esto puede tardar...")

    # Usamos un iterador para no sobrecargar la memoria
    chunk_size_rows = 10000 
    iter_csv = pd.read_csv(original_filepath, usecols=columns_to_keep, iterator=True, chunksize=chunk_size_rows)
    
    start_time = time.time()
    for i, chunk in enumerate(iter_csv):
        mode = 'w' if i == 0 else 'a'
        header = True if i == 0 else False
        chunk.to_csv(output_filepath, index=False, mode=mode, header=header)
        if (i+1) % 10 == 0: # Imprime el progreso cada 10 chunks de filas
             print(f"  > Escritos {(i+1) * chunk_size_rows} filas...")
    end_time = time.time()
        
    print(f"¡Dataset filtrado guardado en '{output_filepath}' en {end_time - start_time:.2f} segundos!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python pre_filter_v3.py <archivo_entrada.csv> <archivo_salida.csv>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # --- MEJORA: Definir explícitamente todas las columnas que NUNCA deben ser eliminadas ---
    target_vars = [
        'ADNC', 'Braak', 'Thal', 'CERAD', 'LATE', 'LEWY',
        'percent 6e10 positive area', 'percent AT8 positive area',
        'percent NeuN positive area', 'percent GFAP positive area',
        'percent aSyn positive area', 'percent pTDP43 positive area']
    # Añadimos Donor_ID a esta lista
    essential_vars = ['Donor_ID'] + target_vars

    survivors = aggressive_pre_filtering(input_file, chunk_size=5000, completeness_threshold=0.8, variance_threshold=0.01)
    
    # Combinamos los sobrevivientes con las variables esenciales, asegurando que no haya duplicados
    final_columns = sorted(list(set(survivors + essential_vars)))
    
    create_filtered_dataset(input_file, final_columns, output_file)

