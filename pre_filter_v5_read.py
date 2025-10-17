# pre_filter_parallel_v4_threads.py
import pandas as pd
import sys
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def process_chunk(chunk_cols, filepath, completeness_threshold, variance_threshold):
    """Procesa un solo chunk de columnas en paralelo (thread-safe)."""
    try:
        df_chunk = pd.read_csv(filepath, usecols=chunk_cols, low_memory=False)
        numeric_chunk = df_chunk.select_dtypes(include=np.number)

        if numeric_chunk.empty:
            return []
        
        completeness = (numeric_chunk != 0).sum() / len(numeric_chunk)
        cols_passing_completeness = completeness[completeness >= completeness_threshold].index
        
        if cols_passing_completeness.empty:
            return []

        variances = numeric_chunk[cols_passing_completeness].var()
        cols_passing_variance = variances[variances > variance_threshold].index.tolist()
        
        print(f"  > Chunk procesado. Sobrevivieron {len(cols_passing_variance)} columnas.")
        return cols_passing_variance
    except Exception as e:
        print(f"Error procesando un chunk: {e}")
        return []

def create_filtered_dataset(original_filepath, columns_to_keep, output_filepath):
    """Crea el archivo CSV final solo con las columnas sobrevivientes."""
    print(f"\nCreando el nuevo dataset filtrado con {len(columns_to_keep)} columnas...")
    iter_csv = pd.read_csv(original_filepath, usecols=columns_to_keep, iterator=True, chunksize=10000)
    for i, chunk in enumerate(iter_csv):
        mode = 'w' if i == 0 else 'a'
        header = True if i == 0 else False
        chunk.to_csv(output_filepath, index=False, mode=mode, header=header)
    print(f"¡Dataset filtrado guardado en '{output_filepath}'!")

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("\nUso: python pre_filter_parallel_v4_threads.py <archivo_de_entrada.csv> <archivo_de_salida.csv>\n")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # --- Parámetros configurables ---
    chunk_size = 5000
    completeness_thresh = 0.8
    variance_thresh = 0.01
    num_threads = 20  # Número de hilos Python (E/S y cálculo)
    numpy_threads = 1  # Número de hilos internos por hilo Python

    # --- Control de hilos internos de BLAS/NumPy ---
    os.environ["OMP_NUM_THREADS"] = str(numpy_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(numpy_threads)
    os.environ["MKL_NUM_THREADS"] = str(numpy_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(numpy_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(numpy_threads)

    start_time = time.time()

    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo de salida: {output_file}")
    print(f"Usando {num_threads} hilos de Python y {numpy_threads} hilos internos por proceso.\n")

    # Leer encabezados con tu método eficiente
    headers = get_headers_efficiently(input_file)

    # Crear lista de chunks
    chunks = [headers[i:i+chunk_size] for i in range(0, len(headers), chunk_size)]
    
    surviving_columns = []

    # Procesamiento paralelo con ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_chunk, chunk, input_file, completeness_thresh, variance_thresh): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            result = future.result()
            surviving_columns.extend(result)

    print(f"\nTotal de columnas que pasaron los filtros: {len(surviving_columns)}")

    # Crear dataset filtrado final
    create_filtered_dataset(input_file, surviving_columns, output_file)
    
    print(f"\nTiempo total: {time.time() - start_time:.2f} segundos")
