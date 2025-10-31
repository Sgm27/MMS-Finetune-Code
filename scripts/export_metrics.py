import json
import os
import glob
import argparse
from tensorboard.backend.event_processing import event_accumulator

def export_tensorboard_to_json(log_dir, output_file):
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"Không tìm thấy event files trong {log_dir}")
        return

    all_metrics = {"training_metrics": {}, "validation_metrics": {}}

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        for tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            metric_type = "validation_metrics" if tag.startswith('val_') else "training_metrics"
            all_metrics[metric_type].setdefault(tag, [])
            for e in events:
                all_metrics[metric_type][tag].append({"step": int(e.step), "value": float(e.value)})

    latest_metrics = {"training_metrics": {}, "validation_metrics": {}, "step": 0}
    max_step = 0
    for metric_type in ["training_metrics", "validation_metrics"]:
        for metric_name, values in all_metrics[metric_type].items():
            if values:
                latest = values[-1]
                latest_metrics[metric_type][metric_name] = latest['value']
                max_step = max(max_step, latest['step'])
    latest_metrics["step"] = max_step

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(latest_metrics, f, indent=2, ensure_ascii=False)
    print(f"Đã xuất metrics ra {output_file}")

def get_latest_metrics_summary(metrics_file):
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    step = data.get('step', 0)
    summary = {"latest_training_metrics": {}, "latest_validation_metrics": {}}
    for metric_name, value in data['training_metrics'].items():
        summary['latest_training_metrics'][metric_name] = {"step": step, "value": value}
    for metric_name, value in data['validation_metrics'].items():
        summary['latest_validation_metrics'][metric_name] = {"step": step, "value": value}
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TensorBoard metrics to JSON")
    parser.add_argument("--log_dir", type=str, required=True, help="Đường dẫn tới thư mục TensorBoard logs")
    parser.add_argument("--output", type=str, default="training_metrics.json", help="Tên file JSON xuất ra")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Thư mục log không tồn tại: {args.log_dir}")
    else:
        export_tensorboard_to_json(args.log_dir, args.output)
        if os.path.exists(args.output):
            summary = get_latest_metrics_summary(args.output)
            print("\nLATEST METRICS SUMMARY")
            print("----------------------")
            for metric, data in summary['latest_training_metrics'].items():
                print(f"Train {metric:25s}: {data['value']:.6f} (step {data['step']})")
            for metric, data in summary['latest_validation_metrics'].items():
                print(f"Val   {metric:25s}: {data['value']:.6f} (step {data['step']})")
