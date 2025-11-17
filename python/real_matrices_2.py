#!/usr/bin/env python3
"""
Matrix Explorer - Esplora matrici della SuiteSparse Matrix Collection
Versione migliorata con ricerca di matrici moltiplicabili
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
import argparse
from collections import defaultdict

dest_dir = "../matrices_downloaded/"  # Cartella per le matrici

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

def get_matrix_sparsity_from_metadata(matrix):
    """Calcola la sparsità dai metadati senza scaricare"""
    if hasattr(matrix, 'rows') and hasattr(matrix, 'cols') and hasattr(matrix, 'nnz'):
        total_elements = matrix.rows * matrix.cols
        if total_elements > 0:
            sparsity = 1 - (matrix.nnz / total_elements)
            return sparsity
    return None

def find_multiplicable_matrices(min_sparsity=0.5, max_sparsity=1.0, min_size=100, max_size=10000, max_results=50):
    """
    Trova coppie di matrici moltiplicabili A×B dove A.cols == B.rows
    
    Args:
        min_sparsity: sparsità minima richiesta (0.5 = almeno 50% di zeri)
        max_sparsity: sparsità massima richiesta (1.0 = fino a 100% di zeri)
        min_size: dimensione minima per evitare matrici troppo piccole
        max_size: dimensione massima per evitare matrici troppo grandi
        max_results: numero massimo di coppie da mostrare
    """
    try:
        print("🔍 Ricerca di matrici moltiplicabili in corso...")
        print("⏳ Recupero informazioni da SuiteSparse Matrix Collection...")
        print("-" * 60)
        
        # Ottieni tutte le matrici
        all_matrices = search(limit=3000)
        
        # Filtra matrici per criteri di base
        suitable_matrices = []
        for matrix in all_matrices:
            # Verifica che abbia le informazioni necessarie
            if not (hasattr(matrix, 'rows') and hasattr(matrix, 'cols') and hasattr(matrix, 'nnz')):
                continue
                
            rows, cols, nnz = matrix.rows, matrix.cols, matrix.nnz
            
            # Filtra per dimensioni
            if rows < min_size or cols < min_size or rows > max_size or cols > max_size:
                continue
            
            # Calcola sparsità
            sparsity = get_matrix_sparsity_from_metadata(matrix)
            if sparsity is None or sparsity < min_sparsity or sparsity > max_sparsity:
                continue
                
            # Filtra solo matrici reali (evita complesse se possibile)
            if hasattr(matrix, 'dtype') and 'complex' in str(matrix.dtype).lower():
                continue
                
            suitable_matrices.append({
                'matrix': matrix,
                'rows': rows,
                'cols': cols,
                'nnz': nnz,
                'sparsity': sparsity
            })
        
        print(f"📊 Matrici adatte trovate: {len(suitable_matrices)}")
        print(f"   (sparsità {min_sparsity:.1%}-{max_sparsity:.1%}, dimensioni {min_size}-{max_size})")
        print()
        
        # Crea dizionario per trovare matrici compatibili velocemente
        matrices_by_rows = defaultdict(list)
        for m in suitable_matrices:
            matrices_by_rows[m['rows']].append(m)
        
        # Trova coppie moltiplicabili
        multiplicable_pairs = []
        
        for matrix_a in suitable_matrices:
            cols_a = matrix_a['cols']
            
            # Cerca matrici B con rows == cols_a
            if cols_a in matrices_by_rows:
                for matrix_b in matrices_by_rows[cols_a]:
                    # Evita di moltiplicare una matrice per se stessa
                    if matrix_a['matrix'].id != matrix_b['matrix'].id:
                        
                        # Calcola dimensione risultante
                        result_rows = matrix_a['rows']
                        result_cols = matrix_b['cols']
                        result_size = result_rows * result_cols
                        
                        # Stima sparsità risultante (approssimativa)
                        estimated_result_sparsity = min(matrix_a['sparsity'], matrix_b['sparsity'])
                        
                        multiplicable_pairs.append({
                            'matrix_a': matrix_a,
                            'matrix_b': matrix_b,
                            'result_rows': result_rows,
                            'result_cols': result_cols,
                            'result_size': result_size,
                            'estimated_sparsity': estimated_result_sparsity
                        })
        
        if not multiplicable_pairs:
            print("❌ Nessuna coppia di matrici moltiplicabili trovata con i criteri specificati")
            return []
        
        # Ordina per dimensione risultante (dalle più piccole alle più grandi)
        multiplicable_pairs.sort(key=lambda x: x['result_size'])
        
        # Limita risultati
        if len(multiplicable_pairs) > max_results:
            multiplicable_pairs = multiplicable_pairs[:max_results]
            print(f"📋 Mostrando le prime {max_results} coppie (ordinate per dimensione risultante):")
        else:
            print(f"📋 Trovate {len(multiplicable_pairs)} coppie moltiplicabili:")
        
        print()
        
        # Mostra le coppie
        for i, pair in enumerate(multiplicable_pairs, 1):
            ma = pair['matrix_a']
            mb = pair['matrix_b']
            
            print(f"[{i:2d}] A: {ma['matrix'].name:25s} ({ma['rows']:4d}×{ma['cols']:4d}, sparsità {ma['sparsity']:.3f}) ×")
            print(f"     B: {mb['matrix'].name:25s} ({mb['rows']:4d}×{mb['cols']:4d}, sparsità {mb['sparsity']:.3f})")
            print(f"     → Risultato: ({pair['result_rows']:4d}×{pair['result_cols']:4d}), "
                  f"sparsità stimata: {pair['estimated_sparsity']:.3f}")
            print(f"     📁 Gruppi: A={ma['matrix'].group} | B={mb['matrix'].group}")
            print(f"     🆔 IDs: A={ma['matrix'].id} | B={mb['matrix'].id}")
            print()
        
        return multiplicable_pairs
        
    except Exception as e:
        print(f"❌ Errore nella ricerca: {e}")
        import traceback
        traceback.print_exc()
        return []

def download_matrix_pair(pair_index, multiplicable_pairs):
    """
    Scarica una coppia specifica di matrici moltiplicabili
    """
    try:
        if pair_index < 1 or pair_index > len(multiplicable_pairs):
            print(f"❌ Indice non valido. Scegli un numero tra 1 e {len(multiplicable_pairs)}")
            return False
        
        pair = multiplicable_pairs[pair_index - 1]
        matrix_a_info = pair['matrix_a']
        matrix_b_info = pair['matrix_b']
        
        print(f"📥 Scaricamento coppia [{pair_index}]:")
        print(f"   A: {matrix_a_info['matrix'].name} (ID: {matrix_a_info['matrix'].id})")
        print(f"   B: {matrix_b_info['matrix'].name} (ID: {matrix_b_info['matrix'].id})")
        print()
        
        # Crea directory se non esiste
        os.makedirs(dest_dir, exist_ok=True)
        
        success_count = 0
        
        # Scarica matrice A
        print("⏳ Scaricamento matrice A...")
        try:
            matrix_list_a = fetch(matrix_a_info['matrix'].id)
            if matrix_list_a and len(matrix_list_a) > 0:
                matrix_list_a[0].download(destpath=dest_dir, extract=True)
                print(f"✅ Matrice A scaricata: {matrix_a_info['matrix'].name}")
                success_count += 1
            else:
                print(f"❌ Errore nel recupero matrice A (ID: {matrix_a_info['matrix'].id})")
        except Exception as e:
            print(f"❌ Errore download matrice A: {e}")
        
        # Scarica matrice B
        print("⏳ Scaricamento matrice B...")
        try:
            matrix_list_b = fetch(matrix_b_info['matrix'].id)
            if matrix_list_b and len(matrix_list_b) > 0:
                matrix_list_b[0].download(destpath=dest_dir, extract=True)
                print(f"✅ Matrice B scaricata: {matrix_b_info['matrix'].name}")
                success_count += 1
            else:
                print(f"❌ Errore nel recupero matrice B (ID: {matrix_b_info['matrix'].id})")
        except Exception as e:
            print(f"❌ Errore download matrice B: {e}")
        
        if success_count == 2:
            print(f"\n🎉 Coppia scaricata con successo in: {dest_dir}")
            print(f"   Ora puoi moltiplicare A ({matrix_a_info['rows']}×{matrix_a_info['cols']}) × B ({matrix_b_info['rows']}×{matrix_b_info['cols']})")
            print(f"   Risultato atteso: {pair['result_rows']}×{pair['result_cols']}")
            return True
        elif success_count == 1:
            print(f"\n⚠️  Solo una matrice è stata scaricata con successo")
            return False
        else:
            print(f"\n❌ Nessuna matrice scaricata con successo")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        return False

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
        print(f"📏 DIMENSIONI")
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
            
            # Crea directory se non esiste
            os.makedirs(dest_dir, exist_ok=True)
            
            try:                
                # Scarica la matrice
                matrix_list[0].download(destpath=dest_dir, extract=True)
                
                print(f"✅ Matrice scaricata in: {dest_dir}")
                return {'info': target_matrix, 'matrix': None}
                
            except Exception as e:
                print(f"❌ Errore durante il download: {e}")
                import traceback
                traceback.print_exc()
                return {'info': target_matrix, 'matrix': None}
        
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
            print(f"     📏 Dimensioni: {getattr(matrix, 'rows', 'N/A'):,} x {getattr(matrix, 'cols', 'N/A'):,} | "
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
            print(f"     📏 Dimensioni: {getattr(matrix, 'rows', 'N/A'):,} x {getattr(matrix, 'cols', 'N/A'):,} | "
                  f"🔢 NNZ: {getattr(matrix, 'nnz', 'N/A'):,}")
            if hasattr(matrix, 'kind'):
                print(f"     🏷️ Kind: {getattr(matrix, 'kind', 'N/A')}")
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
            print(f"{i:2d}. 🏷️ {kind:30s} - {count:4d} matrici")
        
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

def analyze_sparsity_distribution(min_size=100, max_size=10000, limit=3000):
    """Analizza la distribuzione della sparsità nelle matrici disponibili"""
    try:
        print("🔍 Analisi distribuzione sparsità in corso...")
        print("⏳ Recupero informazioni da SuiteSparse Matrix Collection...")
        print("-" * 60)
        
        # Ottieni tutte le matrici
        all_matrices = search(limit=limit)
        
        # Raccogli statistiche sparsità
        sparsities = []
        suitable_matrices = []
        
        for matrix in all_matrices:
            # Verifica che abbia le informazioni necessarie
            if not (hasattr(matrix, 'rows') and hasattr(matrix, 'cols') and hasattr(matrix, 'nnz')):
                continue
                
            rows, cols, nnz = matrix.rows, matrix.cols, matrix.nnz
            
            # Filtra per dimensioni se richiesto
            if min_size > 0 and (rows < min_size or cols < min_size):
                continue
            if max_size > 0 and (rows > max_size or cols > max_size):
                continue
            
            # Calcola sparsità
            sparsity = get_matrix_sparsity_from_metadata(matrix)
            if sparsity is not None:
                sparsities.append(sparsity)
                suitable_matrices.append({
                    'name': matrix.name,
                    'id': matrix.id,
                    'group': getattr(matrix, 'group', 'Unknown'),
                    'rows': rows,
                    'cols': cols,
                    'nnz': nnz,
                    'sparsity': sparsity
                })
        
        if not sparsities:
            print("❌ Nessuna matrice valida trovata per l'analisi")
            return None
        
        # Calcola statistiche
        import numpy as np
        sparsities_array = np.array(sparsities)
        
        min_sparsity = np.min(sparsities_array)
        max_sparsity = np.max(sparsities_array)
        mean_sparsity = np.mean(sparsities_array)
        median_sparsity = np.median(sparsities_array)
        std_sparsity = np.std(sparsities_array)
        
        # Percentili
        p25 = np.percentile(sparsities_array, 25)
        p75 = np.percentile(sparsities_array, 75)
        p90 = np.percentile(sparsities_array, 90)
        p95 = np.percentile(sparsities_array, 95)
        p99 = np.percentile(sparsities_array, 99)
        
        print(f"📊 STATISTICHE SPARSITÀ ({len(sparsities)} matrici analizzate)")
        print(f"   Criteri: dimensioni {min_size}-{max_size if max_size > 0 else '∞'}")
        print()
        print(f"   📈 Sparsità minima:  {min_sparsity:.6f} ({min_sparsity*100:.4f}%)")
        print(f"   📈 Sparsità massima: {max_sparsity:.6f} ({max_sparsity*100:.4f}%)")
        print(f"   📊 Media:           {mean_sparsity:.6f} ({mean_sparsity*100:.4f}%)")
        print(f"   📊 Mediana:         {median_sparsity:.6f} ({median_sparsity*100:.4f}%)")
        print(f"   📊 Deviazione std:  {std_sparsity:.6f}")
        print()
        print(f"   📊 PERCENTILI:")
        print(f"      25°: {p25:.6f} ({p25*100:.4f}%)")
        print(f"      75°: {p75:.6f} ({p75*100:.4f}%)")
        print(f"      90°: {p90:.6f} ({p90*100:.4f}%)")
        print(f"      95°: {p95:.6f} ({p95*100:.4f}%)")
        print(f"      99°: {p99:.6f} ({p99*100:.4f}%)")
        print()
        
        # Distribuzione per fasce
        ranges = [
            (0.0, 0.5, "Bassa (0-50%)"),
            (0.5, 0.7, "Media (50-70%)"),
            (0.7, 0.8, "Alta (70-80%)"),
            (0.8, 0.9, "Molto alta (80-90%)"),
            (0.9, 0.95, "Estrema (90-95%)"),
            (0.95, 1.0, "Ultra-sparse (95-100%)")
        ]
        
        print(f"   📊 DISTRIBUZIONE PER FASCE:")
        for min_range, max_range, label in ranges:
            count = np.sum((sparsities_array >= min_range) & (sparsities_array < max_range))
            percentage = (count / len(sparsities)) * 100
            print(f"      {label:20s}: {count:4d} matrici ({percentage:5.1f}%)")
        
        # Top 10 matrici più sparse
        sorted_matrices = sorted(suitable_matrices, key=lambda x: x['sparsity'], reverse=True)
        top_sparse = sorted_matrices[:10]
        
        print(f"\n   🏆 TOP 10 MATRICI PIÙ SPARSE:")
        for i, m in enumerate(top_sparse, 1):
            print(f"      {i:2d}. {m['name']:25s} | Sparsità: {m['sparsity']:.6f} ({m['sparsity']*100:.4f}%) | "
                  f"Dimensioni: {m['rows']}×{m['cols']} | ID: {m['id']}")
        
        return {
            'statistics': {
                'min': min_sparsity,
                'max': max_sparsity,
                'mean': mean_sparsity,
                'median': median_sparsity,
                'std': std_sparsity,
                'percentiles': {'p25': p25, 'p75': p75, 'p90': p90, 'p95': p95, 'p99': p99}
            },
            'matrices': suitable_matrices,
            'top_sparse': top_sparse
        }
        
    except Exception as e:
        print(f"❌ Errore nell'analisi della sparsità: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Gestione argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Matrix Explorer - SuiteSparse Matrix Collection')
    parser.add_argument('--multiplicable', action='store_true', 
                       help='Cerca coppie di matrici moltiplicabili senza scaricarle')
    parser.add_argument('--analyze-sparsity', action='store_true',
                       help='Analizza la distribuzione della sparsità nelle matrici disponibili')
    parser.add_argument('--min-sparsity', type=float, default=0.5,
                       help='Sparsità minima richiesta (default: 0.5)')
    parser.add_argument('--max-sparsity', type=float, default=1.0,
                       help='Sparsità massima richiesta (default: 1.0)')
    parser.add_argument('--min-size', type=int, default=200,
                       help='Dimensione minima matrici (default: 200)')
    parser.add_argument('--max-size', type=int, default=300,
                       help='Dimensione massima matrici (default: 300)')
    parser.add_argument('--max-results', type=int, default=50,
                       help='Numero massimo di coppie da mostrare (default: 50)')
    
    args = parser.parse_args()
    
    if args.analyze_sparsity:
        print("🎯 Modalità analisi distribuzione sparsità")
        print("=" * 60)
        
        result = analyze_sparsity_distribution(
            min_size=args.min_size,
            max_size=args.max_size if args.max_size > 0 else 0
        )
        
        if result:
            print(f"\n💡 SUGGERIMENTI PER LA RICERCA:")
            stats = result['statistics']
            print(f"   Per matrici molto sparse (>90%): --min-sparsity {stats['percentiles']['p90']:.3f}")
            print(f"   Per matrici estremamente sparse (>95%): --min-sparsity {stats['percentiles']['p95']:.3f}")
            print(f"   Per matrici ultra-sparse (>99%): --min-sparsity {stats['percentiles']['p99']:.3f}")
            print(f"   Sparsità massima disponibile: --max-sparsity {stats['max']:.6f}")
        
        return
    
    if args.multiplicable:
        print("🎯 Modalità ricerca matrici moltiplicabili")
        print("=" * 60)
        
        pairs = find_multiplicable_matrices(
            min_sparsity=args.min_sparsity,
            max_sparsity=args.max_sparsity,
            min_size=args.min_size,
            max_size=args.max_size,
            max_results=args.max_results
        )
        
        if pairs:
            while True:
                try:
                    choice = input(f"\n🎯 Scegli una coppia da scaricare (1-{len(pairs)}) o 'q' per uscire: ").strip().lower()
                    
                    if choice == 'q':
                        print("\n👋 Arrivederci!")
                        break
                    
                    pair_num = int(choice)
                    if download_matrix_pair(pair_num, pairs):
                        print("\n🎉 Download completato!")
                        break
                    else:
                        print("\n❌ Download fallito, riprova con un'altra coppia")
                        
                except ValueError:
                    print("❌ Inserisci un numero valido o 'q' per uscire")
                except KeyboardInterrupt:
                    print("\n\n👋 Arrivederci!")
                    break
        return
    
    # Modalità normale (menu interattivo)
    print("🎯 Matrix Explorer - SuiteSparse Matrix Collection")
    print("=" * 60)
    
    while True:
        print("\nOpzioni disponibili:")
        print("1. Cerca matrice per ID")
        print("2. Cerca matrici per nome")
        print("3. Cerca matrici per gruppo")
        print("4. Lista tipi di matrici disponibili")
        print("5. Lista gruppi di matrici disponibili")
        print("6. Cerca matrici moltiplicabili")
        print("7. Analizza distribuzione sparsità")
        print("8. Esci")
        
        choice = input("\nSeleziona un'opzione (1-8): ").strip()
        
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
            print("\n🎯 Ricerca matrici moltiplicabili")
            print("-" * 40)
            
            # Chiedi parametri personalizzati
            try:
                min_sparsity = input("Sparsità minima (default 0.5): ").strip()
                min_sparsity = float(min_sparsity) if min_sparsity else 0.5
                
                max_sparsity = input("Sparsità massima (default 1.0): ").strip()
                max_sparsity = float(max_sparsity) if max_sparsity else 1.0
                
                min_size = input("Dimensione minima (default 100): ").strip()
                min_size = int(min_size) if min_size else 100
                
                max_size = input("Dimensione massima (default 10000): ").strip()
                max_size = int(max_size) if max_size else 10000
                
                max_results = input("Numero max risultati (default 50): ").strip()
                max_results = int(max_results) if max_results else 50
                
            except ValueError:
                print("❌ Parametri non validi, uso valori di default")
                min_sparsity, max_sparsity, min_size, max_size, max_results = 0.5, 1.0, 100, 10000, 50
            
            pairs = find_multiplicable_matrices(min_sparsity, max_sparsity, min_size, max_size, max_results)
            
            if pairs:
                try:
                    choice_pair = input(f"\n🎯 Scegli una coppia da scaricare (1-{len(pairs)}) o 'n' per continuare: ").strip().lower()
                    
                    if choice_pair != 'n':
                        pair_num = int(choice_pair)
                        download_matrix_pair(pair_num, pairs)
                        
                except ValueError:
                    print("❌ Numero non valido")
            
        elif choice == '7':
            print("\n🎯 Analisi distribuzione sparsità")
            print("-" * 40)
            
            try:
                min_size = input("Dimensione minima (default 100): ").strip()
                min_size = int(min_size) if min_size else 100
                
                max_size = input("Dimensione massima (default 10000, 0=nessun limite): ").strip()
                max_size = int(max_size) if max_size else 10000
                
            except ValueError:
                print("❌ Parametri non validi, uso valori di default")
                min_size, max_size = 100, 10000
            
            result = analyze_sparsity_distribution(min_size, max_size if max_size > 0 else 0)
            
            if result:
                print(f"\n💡 SUGGERIMENTI BASATI SUI DATI:")
                stats = result['statistics']
                print(f"   Per matrici molto sparse: --min-sparsity {stats['percentiles']['p90']:.3f}")
                print(f"   Per matrici estremamente sparse: --min-sparsity {stats['percentiles']['p95']:.3f}")
                print(f"   Sparsità massima: --max-sparsity {stats['max']:.6f}")
                
        elif choice == '8':
            print("\n👋 Arrivederci!")
            break
            
        else:
            print("❌ Opzione non valida")

if __name__ == "__main__":
    main()