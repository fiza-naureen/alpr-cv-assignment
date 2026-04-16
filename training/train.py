from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='dataset/ufpr_alpr.yaml',   # your YAML config
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    lrf=0.1,
    patience=10,
    seed=42,
    project='runs/detect',
    name='alpr_v1'
)

# Save the best model path
best_model = 'runs/detect/alpr_v1/weights/best.pt'
print(f"Training complete. Best model saved to {best_model}")
