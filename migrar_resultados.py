#!/usr/bin/env python3
"""
Script de migración: Convierte JSON de resultados antiguos al esquema canónico.

Uso:
    python migrar_resultados.py --all          # Migrar todos los JSONs
    python migrar_resultados.py --duration 10  # Migrar solo 10seg
    python migrar_resultados.py --backup       # Crear respaldos sin migrar
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


def migrate_vggish_result(old_entry):
    """Convierte entrada antigua de vggish al esquema canónico."""
    
    new_entry = {
        "timestamp": old_entry.get("timestamp", datetime.now().isoformat()),
        "model_type": old_entry.get("model_type", "unknown"),
        "backbone": "vggish",
    }
    
    # Mapear configuración
    if "config" in old_entry:
        new_entry["config"] = old_entry["config"]
    
    # Extraer resultados
    if "results" in old_entry:
        results = old_entry["results"]
        if isinstance(results, dict) and "ensemble_results" in results:
            new_entry["ensemble_results"] = results["ensemble_results"]
        elif isinstance(results, dict) and "plate" in results:
            # Convertir formato antiguo directo
            new_entry["ensemble_results"] = results
    
    # Copiar fold_results si existe, renombrar claves de accuracy
    if "fold_results" in old_entry:
        fold_results = old_entry["fold_results"]
        # Renombrar claves antiguas si es necesario
        for fold in fold_results:
            if "acc_plate" in fold:
                fold["accuracy_plate"] = fold.pop("acc_plate")
            if "acc_electrode" in fold:
                fold["accuracy_electrode"] = fold.pop("acc_electrode")
            if "acc_current" in fold:
                fold["accuracy_current"] = fold.pop("acc_current")
        new_entry["fold_results"] = fold_results
    
    # Copiar campos opcionales
    for field in ["fold_best_epochs", "fold_training_times_seconds", 
                  "improvement_vs_individual", "individual_avg", 
                  "system_info", "model_parameters", "data", "training_history"]:
        if field in old_entry:
            new_entry[field] = old_entry[field]
    
    # Generar ID si no existe
    if "id" not in new_entry:
        config = new_entry.get("config", {})
        duration = config.get("segment_duration", config.get("duration", "?"))
        model = new_entry.get("model_type", "unknown")
        fold_count = config.get("n_folds", config.get("k_folds", "?"))
        overlap = config.get("overlap_ratio", config.get("overlap", "?"))
        new_entry["id"] = f"{duration}seg_{fold_count}fold_overlap_{overlap}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return new_entry


def migrate_spectral_result(old_entry):
    """Convierte entrada antigua de spectral-analysis al esquema canónico."""
    
    new_entry = {
        "timestamp": old_entry.get("timestamp", datetime.now().isoformat()),
        "model_type": old_entry.get("model_type", "unknown"),
        "backbone": "spectral-mfcc",
    }
    
    # Mapear configuración
    if "config" in old_entry:
        new_entry["config"] = old_entry["config"]
    
    # Renombrar blind_evaluation → ensemble_results
    if "blind_evaluation" in old_entry:
        be = old_entry["blind_evaluation"]
        ensemble_results = {}
        
        # Mapear tasks directamente
        for task in ["plate", "electrode", "current"]:
            if task in be:
                ensemble_results[task] = be[task]
        
        # Mapear global si existe
        if "global" in be:
            ensemble_results["global_metrics"] = be["global"]
        
        new_entry["ensemble_results"] = ensemble_results
    
    # Copiar fold_results, renombrando claves de accuracy
    if "fold_results" in old_entry:
        fold_results = old_entry["fold_results"]
        for fold in fold_results:
            if "time_seconds" in fold and "fold_training_times_seconds" not in new_entry:
                # Reconstruir fold_training_times_seconds si no existe
                if "fold_training_times_seconds" not in new_entry:
                    new_entry["fold_training_times_seconds"] = []
                new_entry["fold_training_times_seconds"].append(fold.get("time_seconds", 0))
            
            # Renombrar claves antiguas
            if "acc_plate" in fold:
                fold["accuracy_plate"] = fold.pop("acc_plate")
            if "acc_electrode" in fold:
                fold["accuracy_electrode"] = fold.pop("acc_electrode")
            if "acc_current" in fold:
                fold["accuracy_current"] = fold.pop("acc_current")
        
        new_entry["fold_results"] = fold_results
    
    # Copiar timing si existe (antes era timing, ahora es execution_time + training_time)
    if "timing" in old_entry:
        timing = old_entry["timing"]
        new_entry["execution_time"] = {
            "seconds": timing.get("total_seconds", 0),
            "minutes": timing.get("total_minutes", 0),
            "hours": timing.get("total_minutes", 0) / 60,
        }
    
    if "training_time" in old_entry:
        new_entry["training_time"] = old_entry["training_time"]
    
    # Copiar campos opcionales
    for field in ["fold_best_epochs", "improvement_vs_individual", 
                  "individual_avg", "system_info", "model_parameters", 
                  "data", "training_history"]:
        if field in old_entry:
            new_entry[field] = old_entry[field]
    
    # Generar ID si no existe
    if "id" not in new_entry:
        config = new_entry.get("config", {})
        duration = config.get("duration", "?")
        model = new_entry.get("model_type", "unknown")
        fold_count = config.get("n_folds", "?")
        overlap = config.get("overlap", "?")
        new_entry["id"] = f"{duration}seg_{fold_count}fold_overlap_{overlap}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return new_entry


def migrate_json_file(json_path, project_type):
    """Migra un archivo JSON completo.
    
    Args:
        json_path: Ruta al archivo resultados.json
        project_type: 'vggish', 'spectral', o 'yamnet'
    
    Returns:
        Lista con entradas migradas
    """
    print(f"Leyendo {json_path}...")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    migrate_func = {
        "vggish": migrate_vggish_result,
        "spectral": migrate_spectral_result,
        "yamnet": migrate_vggish_result,  # yamnet usa formato similar a vggish
    }.get(project_type, migrate_vggish_result)
    
    migrated = []
    for entry in data:
        try:
            new_entry = migrate_func(entry)
            migrated.append(new_entry)
        except Exception as e:
            print(f"  ⚠ Error al migrar entrada: {e}")
            migrated.append(entry)  # Mantener entrada original si hay error
    
    return migrated


def create_backup(json_path):
    """Crea respaldo automático del archivo original."""
    backup_path = json_path.parent / f"{json_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy(json_path, backup_path)
    print(f"  ✓ Respaldo creado: {backup_path}")
    return backup_path


def process_duration_dir(duration_dir, project_type, create_backup_only=False):
    """Procesa directorio de duración específica."""
    json_path = duration_dir / "resultados.json"
    
    if not json_path.exists():
        return None
    
    print(f"\nProcesando {duration_dir.name}...")
    
    # Crear respaldo
    create_backup(json_path)
    
    if create_backup_only:
        print(f"  (modo respaldo solamente)")
        return None
    
    # Migrar
    migrated = migrate_json_file(json_path, project_type)
    
    # Guardar
    with open(json_path, "w") as f:
        json.dump(migrated, f, indent=2)
    
    print(f"  ✓ Migrado: {len(migrated)} entradas")
    return len(migrated)


def main():
    parser = argparse.ArgumentParser(
        description="Migrar JSON de resultados al esquema canónico"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrar todos los archivos JSON"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Migrar solo duración específica (ej: 10)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Solo crear respaldos, no migrar"
    )
    parser.add_argument(
        "--project",
        choices=["vggish", "spectral", "yamnet"],
        default="spectral",
        help="Proyecto a migrar (default: spectral)"
    )
    args = parser.parse_args()
    
    if args.project == "vggish":
        base_dir = Path("/home/luis/projects/vggish-backbone")
    elif args.project == "yamnet":
        base_dir = Path("/home/luis/projects/yamnet-backbone")
    else:
        base_dir = Path("/home/luis/projects/spectral-analysis")
    
    print(f"\n{'='*60}")
    print(f"Migración de JSON - Proyecto: {args.project}")
    print(f"{'='*60}")
    
    durations = None
    if args.duration:
        durations = [f"{args.duration:02d}seg"]
        print(f"Duración específica: {args.duration}s")
    else:
        # Detectar duraciones disponibles
        durations = sorted([d.name for d in base_dir.iterdir() 
                          if d.is_dir() and d.name.endswith("seg")])
        if args.all:
            print(f"Migrando TODAS las duraciones encontradas: {durations}")
        else:
            print(f"Duraciones disponibles (usa --all para migrar todas): {durations}")
            print("Uso: python migrar_resultados.py --all")
            return
    
    if args.backup:
        print("Modo: Crear respaldos solamente (sin migrar)\n")
    else:
        print("Modo: Migrar esquema\n")
    
    total_entries = 0
    for duration in durations:
        duration_dir = base_dir / duration
        if duration_dir.exists():
            count = process_duration_dir(
                duration_dir, 
                args.project, 
                create_backup_only=args.backup
            )
            if count is not None:
                total_entries += count
    
    print(f"\n{'='*60}")
    if args.backup:
        print(f"Respaldos creados exitosamente")
    else:
        print(f"Migración completada: {total_entries} entradas procesadas")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
