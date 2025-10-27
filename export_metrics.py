"""
Script để xuất training metrics từ TensorBoard logs ra file JSON
Không cần cài đặt TensorBoard - chỉ cần tensorflow!
"""

import json
import os
import glob
from pathlib import Path

def export_tensorboard_to_json(log_dir, output_file):
    """
    Đọc TensorBoard logs và xuất ra file JSON
    
    Args:
        log_dir: Đường dẫn đến thư mục chứa TensorBoard logs
        output_file: Tên file JSON để lưu kết quả
    """
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("❌ Cần cài đặt tensorboard: pip install tensorboard")
        return
    
    # Tìm file event trong log directory
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"❌ Không tìm thấy TensorBoard event files trong {log_dir}")
        print(f"   Đường dẫn tìm kiếm: {log_dir}")
        return
    
    print(f"✓ Tìm thấy {len(event_files)} event file(s)")
    
    all_metrics = {
        "training_metrics": {},
        "validation_metrics": {}
    }
    
    for event_file in event_files:
        print(f"Đang đọc: {os.path.basename(event_file)}")
        
        try:
            # Load event file với tensorboard
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Lấy tất cả scalar tags
            tags = ea.Tags().get('scalars', [])
            
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
                        "step": int(event.step),
                        "value": float(event.value)
                    })
        except Exception as e:
            print(f"  ⚠️  Lỗi khi đọc file: {str(e)}")
            continue
    
    # Chỉ lấy giá trị cuối cùng (latest) của mỗi metric
    latest_metrics = {
        "training_metrics": {},
        "validation_metrics": {},
        "step": 0
    }
    
    # Tìm step cao nhất
    max_step = 0
    for metric_type in ["training_metrics", "validation_metrics"]:
        for metric_name, values in all_metrics[metric_type].items():
            if values:
                latest = values[-1]
                latest_metrics[metric_type][metric_name] = latest['value']
                max_step = max(max_step, latest['step'])
    
    latest_metrics["step"] = max_step
    
    # Lưu ra file JSON (chỉ latest values)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(latest_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Đã xuất metrics ra file: {output_file}")
    print(f"\n📊 Tổng quan metrics:")
    print(f"  - Training metrics: {list(latest_metrics['training_metrics'].keys())}")
    print(f"  - Validation metrics: {list(latest_metrics['validation_metrics'].keys())}")
    print(f"  - Latest step: {latest_metrics['step']}")


def get_latest_metrics_summary(metrics_file):
    """
    Đọc file metrics (đã là latest format)
    """
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    step = data.get('step', 0)
    summary = {
        "latest_training_metrics": {},
        "latest_validation_metrics": {}
    }
    
    # Format lại để hiển thị với step
    for metric_name, value in data['training_metrics'].items():
        summary['latest_training_metrics'][metric_name] = {
            "step": step,
            "value": value
        }
    
    for metric_name, value in data['validation_metrics'].items():
        summary['latest_validation_metrics'][metric_name] = {
            "step": step,
            "value": value
        }
    
    return summary


if __name__ == "__main__":
    # Tự động tìm thư mục runs
    log_dir = "runs"
    
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
