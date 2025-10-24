"""
Script để xuất training metrics từ TensorBoard logs ra file JSON
"""

import json
import os
import glob
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def export_tensorboard_to_json(log_dir, output_file):
    """
    Đọc TensorBoard logs và xuất ra file JSON
    
    Args:
        log_dir: Đường dẫn đến thư mục chứa TensorBoard logs
        output_file: Tên file JSON để lưu kết quả
    """
    
    # Tìm file event trong log directory
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"❌ Không tìm thấy TensorBoard event files trong {log_dir}")
        return
    
    print(f"✓ Tìm thấy {len(event_files)} event file(s)")
    
    all_metrics = {
        "training_metrics": {},
        "validation_metrics": {}
    }
    
    for event_file in event_files:
        print(f"Đang đọc: {event_file}")
        
        # Load event file
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Lấy tất cả scalar tags
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            events = ea.Scalars(tag)
            
            # Phân loại metrics
            if tag.startswith('train_'):
                metric_type = "training_metrics"
            elif tag.startswith('val_'):
                metric_type = "validation_metrics"
            elif tag == 'lr':
                metric_type = "training_metrics"
            else:
                metric_type = "training_metrics"
            
            # Lưu metrics với step và value
            if tag not in all_metrics[metric_type]:
                all_metrics[metric_type][tag] = []
            
            for event in events:
                all_metrics[metric_type][tag].append({
                    "step": event.step,
                    "value": float(event.value)
                })
    
    # Lưu ra file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Đã xuất metrics ra file: {output_file}")
    print(f"\n📊 Tổng quan metrics:")
    print(f"  - Training metrics: {list(all_metrics['training_metrics'].keys())}")
    print(f"  - Validation metrics: {list(all_metrics['validation_metrics'].keys())}")


def get_latest_metrics_summary(metrics_file):
    """
    Lấy giá trị metrics mới nhất
    """
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = {
        "latest_training_metrics": {},
        "latest_validation_metrics": {}
    }
    
    # Lấy giá trị cuối cùng của mỗi metric
    for metric_name, values in data['training_metrics'].items():
        if values:
            latest = values[-1]
            summary['latest_training_metrics'][metric_name] = {
                "step": latest['step'],
                "value": latest['value']
            }
    
    for metric_name, values in data['validation_metrics'].items():
        if values:
            latest = values[-1]
            summary['latest_validation_metrics'][metric_name] = {
                "step": latest['step'],
                "value": latest['value']
            }
    
    return summary


if __name__ == "__main__":
    # Cấu hình
    config_file = "training_config_examples/finetune_mms_vie.json"
    
    # Đọc config để lấy output_dir
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        output_dir = config.get('output_dir', '/vits_finetuned_vie')
        
        # Nếu là đường dẫn tương đối hoặc bắt đầu bằng /, coi như là thư mục local
        if output_dir.startswith('/'):
            output_dir = output_dir.lstrip('/')
        
        log_dir = os.path.join(output_dir, "runs")
    else:
        # Mặc định
        output_dir = "vits_finetuned_vie"
        log_dir = os.path.join(output_dir, "runs")
    
    # Kiểm tra xem thư mục log có tồn tại không
    if not os.path.exists(log_dir):
        print(f"⚠️  Thư mục log chưa tồn tại: {log_dir}")
        print(f"    Thư mục này sẽ được tạo sau khi bắt đầu training.")
        print(f"    Vui lòng chạy lại script này sau khi training bắt đầu.")
    else:
        # Xuất metrics
        output_file = "training_metrics.json"
        export_tensorboard_to_json(log_dir, output_file)
        
        # In summary
        if os.path.exists(output_file):
            summary = get_latest_metrics_summary(output_file)
            
            print("\n" + "="*60)
            print("📈 LATEST METRICS SUMMARY")
            print("="*60)
            
            print("\n🔹 Training Metrics (Latest):")
            for metric, data in summary['latest_training_metrics'].items():
                print(f"  {metric:30s} = {data['value']:.6f} (step {data['step']})")
            
            if summary['latest_validation_metrics']:
                print("\n🔹 Validation Metrics (Latest):")
                for metric, data in summary['latest_validation_metrics'].items():
                    print(f"  {metric:30s} = {data['value']:.6f} (step {data['step']})")
