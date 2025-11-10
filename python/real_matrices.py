#!/usr/bin/env python3
"""
Matrix Explorer - Esplora matrici della SuiteSparse Matrix Collection
Versione migliorata con gestione download più robusta
"""

from ssgetpy import search, fetch
import numpy as np
from scipy import sparse
import sys
import tempfile
import os
import glob
from scipy.io import loadmat
import shutil

dest_dir="./" #SCRIVERE FOLDER PER MATRICI
def format_size(num_bytes):
    """Formatta la dimensione in bytes in modo leggibile"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def calculate_sparsity(matrix):
    """Calcola il grado di sparsità della matrice"""
    if sparse.issparse(matrix):
        total_elements = matrix.shape[0] * matrix.shape[1]
        non_zero_elements = matrix.nnz
    else:
        total_elements = matrix.size
        non_zero_elements = np.count_nonzero(matrix)
    
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity, non_zero_elements, total_elements

def find_matrix_files(start_path="."):
    """Trova tutti i file di matrice possibili"""
    extensions = ['*.mat', '*.mtx', '*.txt', '*.dat']
    found_files = []
    
    for ext in extensions:
        pattern = os.path.join(start_path, '**', ext)
        files = glob.glob(pattern, recursive=True)
        found_files.extend(files)
    
    return found_files

def load_matrix_from_file(file_path):
    """Carica una matrice da vari formati di file"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.mat':
            return load_mat_file(file_path)
        elif file_ext == '.mtx':
            return load_mtx_file(file_path)
        else:
            # Prova a leggere come file di testo
            return load_text_file(file_path)
    except Exception as e:
        print(f"❌ Errore nel caricamento di {file_path}: {e}")
        return None

def load_mat_file(file_path):
    """Carica una matrice da file .mat"""
    mat_data = loadmat(file_path)
    
    # Cerca la matrice nei vari campi possibili
    possible_keys = ['Problem', 'A', 'data', 'matrix']
    
    for key in possible_keys:
        if key in mat_data:
            data = mat_data[key]
            
            if key == 'Problem' and isinstance(data, np.ndarray) and data.size > 0:
                problem = data[0,0]
                if hasattr(problem, 'dtype') and 'A' in problem.dtype.names:
                    return problem['A'][0,0]
            elif sparse.issparse(data) or isinstance(data, np.ndarray):
                return data
    
    # Se non trova nei campi standard, cerca qualsiasi matrice
    for key, value in mat_data.items():
        if not key.startswith('__'):
            if sparse.issparse(value) or (isinstance(value, np.ndarray) and value.ndim >= 2):
                return value
    
    return None

def load_mtx_file(file_path):
    """Carica una matrice da file .mtx (Matrix Market format)"""
    try:
        from scipy.io import mmread
        return mmread(file_path)
    except ImportError:
        print("❌ scipy.io.mmread non disponibile")
        return None

def load_text_file(file_path):
    """Prova a caricare una matrice da file di testo"""
    try:
        return np.loadtxt(file_path)
    except:
        return None

def display_matrix_info(matrix_id):
    """Visualizza informazioni dettagliate sulla matrice"""
    try:
        print(f"🔍 Ricerca matrice con ID: {matrix_id}")
        print("-" * 60)
        
        # Usa fetch direttamente con l'ID per ottenere i metadati
        try:
            matrix_list = fetch(matrix_id)
            if not matrix_list or len(matrix_list) == 0:
                print(f"❌ Nessuna matrice trovata con ID: {matrix_id}")
                return None
            
            target_matrix = matrix_list[0]
        except Exception as e:
            print(f"❌ Errore nel recupero della matrice con ID {matrix_id}: {e}")
            return None
        
        # Informazioni di base
        print(f"📋 INFORMAZIONI GENERALI")
        print(f"   Nome: {getattr(target_matrix, 'name', 'N/A')}")
        print(f"   ID: {getattr(target_matrix, 'id', 'N/A')}")
        print(f"   Gruppo: {getattr(target_matrix, 'group', 'N/A')}")
        print(f"   Kind: {getattr(target_matrix, 'kind', 'N/A')}")
        print()
        
        # Dimensioni
        print(f"📐 DIMENSIONI")
        print(f"   Righe: {getattr(target_matrix, 'rows', 'N/A'):,}")
        print(f"   Colonne: {getattr(target_matrix, 'cols', 'N/A'):,}")
        print(f"   Elementi non-zero: {getattr(target_matrix, 'nnz', 'N/A'):,}")
        print()
        
        # Proprietà della matrice
        print(f"🔧 PROPRIETÀ")
        print(f"   Tipo: {getattr(target_matrix, 'dtype', 'N/A')}")
        if hasattr(target_matrix, 'psym'):
            print(f"   Simmetria Pattern: {getattr(target_matrix, 'psym', 'N/A')}")
        if hasattr(target_matrix, 'nsym'):
            print(f"   Simmetria Numerica: {getattr(target_matrix, 'nsym', 'N/A')}")
        print(f"   Definita positiva: {'Sì' if getattr(target_matrix, 'isspd', False) else 'No'}")
        print()
        
        # Calcola sparsity dai metadati
        if hasattr(target_matrix, 'rows') and hasattr(target_matrix, 'cols') and hasattr(target_matrix, 'nnz'):
            total_elements = target_matrix.rows * target_matrix.cols
            sparsity = 1 - (target_matrix.nnz / total_elements)
            print(f"📊 SPARSITÀ")
            print(f"   Grado di sparsità: {sparsity:.6f} ({sparsity*100:.4f}%)")
            print(f"   Densità: {(1-sparsity)*100:.4f}%")
            print()
        
        # Chiedi se scaricare la matrice
        response = input("🤔 Vuoi scaricare e analizzare la matrice? (s/n): ").lower().strip()
        
        if response == 's' or response == 'sì':
            print("\n⏳ Scaricamento matrice in corso...")
            
            # Salva directory corrente
            original_dir = os.getcwd()
            
            try:                
                # Scarica la matrice
                matrix_list[0].download(destpath=dest_dir,extract=True)
                
                # Cerca file di matrice
                print("🔍 Ricerca file di matrice...")
                found_files = find_matrix_files()
                
                if not found_files:
                    print("❌ Nessun file di matrice trovato")
                    print("📂 Contenuto directory:")
                    for root, dirs, files in os.walk("."):
                        level = root.replace(".", "").count(os.sep)
                        indent = " " * 2 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        sub_indent = " " * 2 * (level + 1)
                        for file in files:
                            print(f"{sub_indent}{file}")
                    return {'info': target_matrix, 'matrix': None}
                
                print(f"📋 File trovati: {found_files}")
                
                # Prova a caricare ogni file trovato
                actual_matrix = None
                for file_path in found_files:
                    print(f"🔄 Tentativo di caricamento: {file_path}")
                    matrix = load_matrix_from_file(file_path)
                    if matrix is not None:
                        actual_matrix = matrix
                        print(f"✅ Matrice caricata da: {file_path}")
                        break
                
                if actual_matrix is None:
                    print("❌ Impossibile caricare la matrice da nessun file")
                    return {'info': target_matrix, 'matrix': None}
                
                # Analisi dettagliata
                print("\n🔬 ANALISI DETTAGLIATA")
                print(f"🔍 Tipo matrice: {type(actual_matrix)}")
                
                sparsity, nnz, total = calculate_sparsity(actual_matrix)
                print(f"   Forma matrice: {actual_matrix.shape}")
                print(f"   Tipo dati: {actual_matrix.dtype}")
                print(f"   Elementi totali: {total:,}")
                print(f"   Elementi non-zero: {nnz:,}")
                print(f"   Sparsità reale: {sparsity:.6f} ({sparsity*100:.4f}%)")
                
                if sparse.issparse(actual_matrix):
                    print(f"   Formato sparse: {type(actual_matrix).__name__}")
                    if hasattr(actual_matrix, 'data'):
                        memory_used = actual_matrix.data.nbytes
                        if hasattr(actual_matrix, 'indices'):
                            memory_used += actual_matrix.indices.nbytes
                        if hasattr(actual_matrix, 'indptr'):
                            memory_used += actual_matrix.indptr.nbytes
                        print(f"   Memoria utilizzata: {format_size(memory_used)}")
                else:
                    print(f"   Memoria utilizzata: {format_size(actual_matrix.nbytes)}")
                
                # Statistiche sui valori
                if sparse.issparse(actual_matrix):
                    data = actual_matrix.data
                else:
                    data = actual_matrix[actual_matrix != 0]
                
                if len(data) > 0:
                    print(f"\n📈 STATISTICHE VALORI")
                    print(f"   Valore minimo: {np.min(data):.6e}")
                    print(f"   Valore massimo: {np.max(data):.6e}")
                    print(f"   Valore medio: {np.mean(data):.6e}")
                    print(f"   Deviazione standard: {np.std(data):.6e}")
                
                # Visualizza alcuni elementi se la matrice è piccola
                if actual_matrix.shape[0] <= 10 and actual_matrix.shape[1] <= 10:
                    print(f"\n👁️  VISUALIZZAZIONE MATRICE (piccola):")
                    if sparse.issparse(actual_matrix):
                        print(actual_matrix.toarray())
                    else:
                        print(actual_matrix)
                
                return {'info': target_matrix, 'matrix': actual_matrix}
                
            except Exception as e:
                print(f"❌ Errore durante il download: {e}")
                import traceback
                traceback.print_exc()
                return {'info': target_matrix, 'matrix': None}
            finally:
                # Ripristina directory originale
                os.chdir(original_dir)
                # Pulisci directory temporanea
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Directory temporanea rimossa: {temp_dir}")
                except:
                    pass
        
        return {'info': target_matrix, 'matrix': None}
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return None

def search_matrices_by_name(name_pattern, limit=20):
    """Cerca matrici per nome o pattern nel nome"""
    try:
        print(f"🔍 Ricerca matrici con nome contenente: '{name_pattern}'")
        print("-" * 60)
        
        # Ottieni tutte le matrici
        all_matrices = search(limit=3000)
        
        # Filtra per nome
        matching_matrices = []
        for matrix in all_matrices:
            if hasattr(matrix, 'name') and name_pattern.lower() in matrix.name.lower():
                matching_matrices.append(matrix)
        
        if not matching_matrices:
            print(f"❌ Nessuna matrice trovata con nome contenente: '{name_pattern}'")
            return []
        
        # Mostra risultati (limitati)
        print(f"📋 Trovate {len(matching_matrices)} matrici. Mostrando le prime {min(limit, len(matching_matrices))}:")
        print()
        
        displayed_matrices = matching_matrices[:limit]
        for i, matrix in enumerate(displayed_matrices, 1):
            print(f"{i:2d}. 🆔 ID: {getattr(matrix, 'id', 'N/A'):4d} | "
                  f"📛 Nome: {getattr(matrix, 'name', 'N/A'):30s} | "
                  f"📁 Gruppo: {getattr(matrix, 'group', 'N/A'):20s}")
            print(f"     📐 Dimensioni: {getattr(matrix, 'rows', 'N/A'):,} x {getattr(matrix, 'cols', 'N/A'):,} | "
                  f"🔢 NNZ: {getattr(matrix, 'nnz', 'N/A'):,}")
            print()
        
        if len(matching_matrices) > limit:
            print(f"... e altre {len(matching_matrices) - limit} matrici")
        
        return matching_matrices
        
    except Exception as e:
        print(f"❌ Errore nella ricerca: {e}")
        return []

def search_matrices_by_group(group_name, limit=20):
    """Cerca matrici per gruppo"""
    try:
        print(f"🔍 Ricerca matrici nel gruppo: '{group_name}'")
        print("-" * 60)
        
        # Ottieni tutte le matrici
        all_matrices = search(limit=3000)
        
        # Filtra per gruppo
        matching_matrices = []
        for matrix in all_matrices:
            if hasattr(matrix, 'group') and group_name.lower() in matrix.group.lower():
                matching_matrices.append(matrix)
        
        if not matching_matrices:
            print(f"❌ Nessuna matrice trovata nel gruppo: '{group_name}'")
            return []
        
        # Mostra risultati (limitati)
        print(f"📋 Trovate {len(matching_matrices)} matrici. Mostrando le prime {min(limit, len(matching_matrices))}:")
        print()
        
        displayed_matrices = matching_matrices[:limit]
        for i, matrix in enumerate(displayed_matrices, 1):
            print(f"{i:2d}. 🆔 ID: {getattr(matrix, 'id', 'N/A'):4d} | "
                  f"📛 Nome: {getattr(matrix, 'name', 'N/A'):30s} | "
                  f"📁 Gruppo: {getattr(matrix, 'group', 'N/A'):20s}")
            print(f"     📐 Dimensioni: {getattr(matrix, 'rows', 'N/A'):,} x {getattr(matrix, 'cols', 'N/A'):,} | "
                  f"🔢 NNZ: {getattr(matrix, 'nnz', 'N/A'):,}")
            if hasattr(matrix, 'kind'):
                print(f"     🏷️  Kind: {getattr(matrix, 'kind', 'N/A')}")
            print()
        
        if len(matching_matrices) > limit:
            print(f"... e altre {len(matching_matrices) - limit} matrici")
        
        return matching_matrices
        
    except Exception as e:
        print(f"❌ Errore nella ricerca: {e}")
        return []

def list_available_kinds():
    """Lista tutti i tipi (kind) di matrici disponibili"""
    try:
        print("🔍 Recupero informazioni sui tipi di matrici...")
        print("-" * 60)
        
        # Ottieni tutte le matrici
        all_matrices = search(limit=3000)
        
        # Raccogli tutti i kind
        kinds = {}
        for matrix in all_matrices:
            if hasattr(matrix, 'kind'):
                kind = getattr(matrix, 'kind', 'Unknown')
                if kind in kinds:
                    kinds[kind] += 1
                else:
                    kinds[kind] = 1
        
        # Ordina per frequenza
        sorted_kinds = sorted(kinds.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📊 Tipi di matrici disponibili ({len(sorted_kinds)} tipi diversi):")
        print()
        
        for i, (kind, count) in enumerate(sorted_kinds, 1):
            print(f"{i:2d}. 🏷️  {kind:30s} - {count:4d} matrici")
        
        print(f"\n📈 Totale matrici analizzate: {len(all_matrices)}")
        
        return sorted_kinds
        
    except Exception as e:
        print(f"❌ Errore nel recupero dei tipi: {e}")
        return []

def list_available_groups():
    """Lista tutti i gruppi di matrici disponibili"""
    try:
        print("🔍 Recupero informazioni sui gruppi di matrici...")
        print("-" * 60)
        
        # Ottieni tutte le matrici
        all_matrices = search(limit=3000)
        
        # Raccogli tutti i gruppi
        groups = {}
        for matrix in all_matrices:
            if hasattr(matrix, 'group'):
                group = getattr(matrix, 'group', 'Unknown')
                if group in groups:
                    groups[group] += 1
                else:
                    groups[group] = 1
        
        # Ordina alfabeticamente
        sorted_groups = sorted(groups.items())
        
        print(f"📊 Gruppi di matrici disponibili ({len(sorted_groups)} gruppi diversi):")
        print()
        
        for i, (group, count) in enumerate(sorted_groups, 1):
            print(f"{i:3d}. 📁 {group:30s} - {count:4d} matrici")
        
        print(f"\n📈 Totale matrici analizzate: {len(all_matrices)}")
        
        return sorted_groups
        
    except Exception as e:
        print(f"❌ Errore nel recupero dei gruppi: {e}")
        return []

def main():
    print("🎯 Matrix Explorer - SuiteSparse Matrix Collection")
    print("=" * 60)
    
    while True:
        print("\nOpzioni disponibili:")
        print("1. Cerca matrice per ID")
        print("2. Cerca matrici per nome")
        print("3. Cerca matrici per gruppo")
        print("4. Lista tipi di matrici disponibili")
        print("5. Lista gruppi di matrici disponibili")
        print("6. Esci")
        
        choice = input("\nSeleziona un'opzione (1-6): ").strip()
        
        if choice == '1':
            try:
                matrix_id = int(input("\n🔢 Inserisci l'ID della matrice: "))
                result = display_matrix_info(matrix_id)
                if result and result['matrix'] is not None:
                    print(f"\n✅ Matrice caricata con successo!")
                    print(f"   Puoi ora utilizzare result['matrix'] per ulteriori analisi")
               
            except ValueError:
                print("❌ Inserisci un ID numerico valido")
                
        elif choice == '2':
            name_pattern = input("\n🔤 Inserisci il nome (o parte del nome) della matrice: ").strip()
            if name_pattern:
                matches = search_matrices_by_name(name_pattern)
                if matches:
                    try:
                        selected = input(f"\n🎯 Vuoi vedere i dettagli di una matrice? Inserisci l'ID (o 'n' per continuare): ").strip()
                        if selected.lower() != 'n':
                            matrix_id = int(selected)
                            display_matrix_info(matrix_id)
                    except ValueError:
                        print("❌ ID non valido")
            else:
                print("❌ Inserisci un nome valido")
                
        elif choice == '3':
            group_name = input("\n📁 Inserisci il nome del gruppo: ").strip()
            if group_name:
                matches = search_matrices_by_group(group_name)
                if matches:
                    try:
                        selected = input(f"\n🎯 Vuoi vedere i dettagli di una matrice? Inserisci l'ID (o 'n' per continuare): ").strip()
                        if selected.lower() != 'n':
                            matrix_id = int(selected)
                            display_matrix_info(matrix_id)
                    except ValueError:
                        print("❌ ID non valido")
            else:
                print("❌ Inserisci un nome di gruppo valido")
                
        elif choice == '4':
            list_available_kinds()
                
        elif choice == '5':
            list_available_groups()
            
        elif choice == '6':
            print("\n👋 Arrivederci!")
            break
            
        else:
            print("❌ Opzione non valida")

if __name__ == "__main__":
    main()