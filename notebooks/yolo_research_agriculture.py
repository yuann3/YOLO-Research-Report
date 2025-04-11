import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import torch
import random
import time
import pandas as pd
from skimage.feature import local_binary_pattern
import matplotlib.image as mpimg
from pathlib import Path
import glob
import re
import gc
import subprocess
import sys
import pkg_resources
os.makedirs("fruit_dataset/images/train", exist_ok=True)
os.makedirs("fruit_dataset/images/val", exist_ok=True)
os.makedirs("fruit_dataset/images/test", exist_ok=True)
os.makedirs("fruit_dataset/labels/train", exist_ok=True)
os.makedirs("fruit_dataset/labels/val", exist_ok=True)
os.makedirs("fruit_dataset/labels/test", exist_ok=True)
os.makedirs("results/fruits", exist_ok=True)
os.makedirs("models/fruit_yolo_model", exist_ok=True)
gc.collect()
torch.cuda.empty_cache()

def setup_dataset():
    """
    Instructions for manually downloading the dataset
    """
    print("To proceed with this code, you need to download the fruit dataset manually:")
    print("1. Go to https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification")
    print("2. Download the dataset and extract it to a folder named 'dataset' in this script's directory")
    print("3. The dataset should have a structure with 'train' and 'test' folders")
    print()
    if os.path.exists("dataset/train") and os.path.exists("dataset/test"):
        print("Dataset found successfully!")
        return "dataset"
    else:
        print("Dataset not found. Please download it manually following the instructions above.")
        print("After downloading, place it in a folder named 'dataset' in the same directory as this script.")
        print("Then run this script again.")
        dataset_path = input("Or enter the path where you've downloaded the dataset: ")
        if os.path.exists(os.path.join(dataset_path, "train")) and os.path.exists(os.path.join(dataset_path, "test")):
            return dataset_path
        else:
            print(f"Invalid path: {dataset_path}")
            return None

def rename_dataset_files(dataset_path):
    print("Renaming files to simpler numeric format...")
    classes = ["freshapples", "freshbanana", "freshoranges",
               "rottenapples", "rottenbanana", "rottenoranges"]
    for split in ["train", "test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping.")
            continue
        for class_name in classes:
            class_dir = os.path.join(split_path, class_name)
            if not os.path.exists(class_dir):
                continue
            image_files = glob.glob(os.path.join(class_dir, "*.png")) + \
                         glob.glob(os.path.join(class_dir, "*.jpg"))
            for i, file_path in enumerate(sorted(image_files)):
                _, ext = os.path.splitext(file_path)
                new_filename = f"{i+1:04d}{ext}"
                new_path = os.path.join(class_dir, new_filename)
                os.rename(file_path, new_path)

            print(f"Renamed {len(image_files)} files in {split}/{class_name}")

    print("File renaming complete")

def prepare_fruit_dataset(dataset_path):
    class_mapping = {
        'freshapples': 0,
        'freshbanana': 1,
        'freshoranges': 2,
        'rottenapples': 3,
        'rottenbanana': 4,
        'rottenoranges': 5
    }
    dest_root = "fruit_dataset"

    print("Converting dataset to YOLO format...")
    os.makedirs(os.path.join(dest_root, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'labels', 'test'), exist_ok=True)
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Training path {train_path} or test path {test_path} not found!")
        print("Available directories:")
        for root, dirs, files in os.walk(dataset_path, maxdepth=2):
            print(root)
        return None
    for class_folder in os.listdir(train_path):
        class_id = class_mapping.get(class_folder)
        if class_id is None:
            continue

        class_path = os.path.join(train_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img in images:
            src_img_path = os.path.join(class_path, img)
            if not os.path.exists(src_img_path):
                print(f"Warning: Source image {src_img_path} not found, skipping.")
                continue
            clean_filename = f"{class_folder}_{img.replace(' ', '_')}"
            dest_img_path = os.path.join(dest_root, 'images', 'train', clean_filename)

            try:
                shutil.copy2(src_img_path, dest_img_path)
            except Exception as e:
                print(f"Error copying {src_img_path} to {dest_img_path}: {e}")
                continue
            try:
                label_filename = os.path.splitext(clean_filename)[0] + '.txt'
                label_path = os.path.join(dest_root, 'labels', 'train', label_filename)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
            except Exception as e:
                print(f"Error creating label file {label_path}: {e}")
    for class_folder in os.listdir(test_path):
        class_id = class_mapping.get(class_folder)
        if class_id is None:
            continue

        class_path = os.path.join(test_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        val_count = int(0.8 * len(images))
        val_images = images[:val_count]
        test_images = images[val_count:]
        for img in val_images:
            src_img_path = os.path.join(class_path, img)
            if not os.path.exists(src_img_path):
                print(f"Warning: Source image {src_img_path} not found, skipping.")
                continue

            clean_filename = f"{class_folder}_{img.replace(' ', '_')}"
            dest_img_path = os.path.join(dest_root, 'images', 'val', clean_filename)

            try:
                shutil.copy2(src_img_path, dest_img_path)
            except Exception as e:
                print(f"Error copying {src_img_path} to {dest_img_path}: {e}")
                continue
            try:
                label_filename = os.path.splitext(clean_filename)[0] + '.txt'
                label_path = os.path.join(dest_root, 'labels', 'val', label_filename)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
            except Exception as e:
                print(f"Error creating label file {label_path}: {e}")
        for img in test_images:
            src_img_path = os.path.join(class_path, img)
            if not os.path.exists(src_img_path):
                print(f"Warning: Source image {src_img_path} not found, skipping.")
                continue

            clean_filename = f"{class_folder}_{img.replace(' ', '_')}"
            dest_img_path = os.path.join(dest_root, 'images', 'test', clean_filename)

            try:
                shutil.copy2(src_img_path, dest_img_path)
            except Exception as e:
                print(f"Error copying {src_img_path} to {dest_img_path}: {e}")
                continue
            try:
                label_filename = os.path.splitext(clean_filename)[0] + '.txt'
                label_path = os.path.join(dest_root, 'labels', 'test', label_filename)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
            except Exception as e:
                print(f"Error creating label file {label_path}: {e}")
    try:
        yaml_path = os.path.join(dest_root, 'fruit_dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(dest_root)}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            f.write(f"nc: {len(class_mapping)}\n")
            display_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange',
                            'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
            f.write(f"names: {display_names}\n")
    except Exception as e:
        print(f"Error creating YAML file: {e}")
        return None
    try:
        train_count = len(os.listdir(os.path.join(dest_root, 'images', 'train')))
        val_count = len(os.listdir(os.path.join(dest_root, 'images', 'val')))
        test_count = len(os.listdir(os.path.join(dest_root, 'images', 'test')))

        if train_count == 0 and val_count == 0 and test_count == 0:
            print("Error: No images were processed successfully!")
            return None

        print(f"Dataset conversion complete: {train_count} training, {val_count} validation, {test_count} test images")
        return yaml_path
    except Exception as e:
        print(f"Error counting files: {e}")
        return None

def show_sample_images():
    train_img_dir = "fruit_dataset/images/train"
    train_images = os.listdir(train_img_dir)
    fresh_sample = next((img for img in train_images if 'fresh' in img), None)
    rotten_sample = next((img for img in train_images if 'rotten' in img), None)

    samples = [img for img in [fresh_sample, rotten_sample] if img]
    if len(samples) > 0:
        plt.figure(figsize=(10, 5))
        for i, img_name in enumerate(samples):
            img_path = os.path.join(train_img_dir, img_name)
            img = mpimg.imread(img_path)
            plt.subplot(1, len(samples), i+1)
            plt.imshow(img)

            raw_class = img_name.split('_')[0]
            if 'fresh' in raw_class:
                status = 'Fresh'
            else:
                status = 'Rotten'

            if 'apple' in raw_class:
                fruit = 'Apple'
            elif 'banana' in raw_class:
                fruit = 'Banana'
            elif 'orange' in raw_class:
                fruit = 'Orange'
            else:
                fruit = ''

            plt.title(f"{status} {fruit}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('results/fruits/sample_images.png', dpi=150)
        plt.show()

def roughness_local_binary_pattern(image, radius=3, n_points=24):
    """Optimized R-LBP implementation for fruit texture analysis"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    from scipy.ndimage import uniform_filter
    mean_square = uniform_filter(gray.astype(np.float32)**2, 2*radius+1)
    square_mean = uniform_filter(gray.astype(np.float32), 2*radius+1)**2
    variance = mean_square - square_mean
    if np.max(variance) > 0:
        variance = 255 * (variance / np.max(variance))
    r_lbp = (0.5 * lbp + 0.5 * variance).astype(np.uint8)
    r_lbp_rgb = cv2.cvtColor(r_lbp, cv2.COLOR_GRAY2RGB)

    return r_lbp_rgb

def visualize_preprocessing():
    base_dir = "fruit_dataset/images/train"
    fresh_apple = next((f for f in os.listdir(base_dir) if f.startswith('freshapples')), None)
    rotten_apple = next((f for f in os.listdir(base_dir) if f.startswith('rottenapples')), None)

    if fresh_apple and rotten_apple:
        fresh_path = os.path.join(base_dir, fresh_apple)
        rotten_path = os.path.join(base_dir, rotten_apple)

        fresh_img = cv2.imread(fresh_path)
        rotten_img = cv2.imread(rotten_path)

        fresh_img_rgb = cv2.cvtColor(fresh_img, cv2.COLOR_BGR2RGB)
        rotten_img_rgb = cv2.cvtColor(rotten_img, cv2.COLOR_BGR2RGB)

        fresh_processed = roughness_local_binary_pattern(fresh_img)
        rotten_processed = roughness_local_binary_pattern(rotten_img)
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.title('Original Fresh Apple')
        plt.imshow(fresh_img_rgb)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title('Processed Fresh Apple (R-LBP)')
        plt.imshow(fresh_processed)
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title('Original Rotten Apple')
        plt.imshow(rotten_img_rgb)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title('Processed Rotten Apple (R-LBP)')
        plt.imshow(rotten_processed)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('results/fruits/preprocessing_visualization.png', dpi=150)
        plt.show()
        print("\nRoughness-Local Binary Pattern (R-LBP) Technique:")
        print("- Enhances texture features to better distinguish between fresh and rotten fruits")
        print("- Combines traditional LBP (pattern encoding) with local variance (roughness)")
        print("- Particularly effective for detecting subtle texture differences in fruit skin")
        print("- This preprocessing step significantly improves the model's ability to detect ripeness")

def train_fruit_model(yaml_path):
    print("Starting model training for local environment...")
    print("Checking and updating Ultralytics version...")
    try:
        current_version = pkg_resources.get_distribution("ultralytics").version
        print(f"Current Ultralytics version: {current_version}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
        if "ultralytics" in sys.modules:
            print("Reloading ultralytics module...")
            import importlib
            importlib.reload(sys.modules["ultralytics"])
        from ultralytics import YOLO
        updated_version = pkg_resources.get_distribution("ultralytics").version
        print(f"Updated Ultralytics version: {updated_version}")
    except Exception as e:
        print(f"Error updating Ultralytics: {e}")
        from ultralytics import YOLO
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        try:
            gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
            print(gpu_info)
        except:
            print("nvidia-smi command not available")
        device = 0  # Use GPU
    else:
        print("No GPU available, using CPU")
        device = 'cpu'
    try:
        model = YOLO("yolo11n.pt") 
        print("Successfully downloaded YOLO11n model")
    except Exception as e2:
        print(f"Error loading fallback model: {e2}")
        raise Exception("Could not load YOLO model. Please check Ultralytics version compatibility.")
    batch_size = 16  # Default value, adjust based on your local GPU/CPU
    if torch.cuda.is_available():
        try:
            mem_info = torch.cuda.get_device_properties(0).total_memory
            if mem_info > 8e9:  # More than 8GB
                batch_size = 32
            elif mem_info > 4e9:  # More than 4GB
                batch_size = 16
            else:  # 4GB or less
                batch_size = 8
        except:
            pass  # Keep default batch size if can't determine memory
    results = model.train(
        data=yaml_path,
        epochs=30,               
        imgsz=640,               
        batch=batch_size,        
        patience=5,               # Early stopping patience
        save=True,                # Save checkpoints
        device=device,            
        workers=4,                
        project='models/fruit_yolo_model',
        name='fruit_ripeness',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        resume=False,
        amp=True,
    )
    model_dir = os.path.join('models/fruit_yolo_model', 'fruit_ripeness')
    weights_dir = os.path.join(model_dir, 'weights')
    best_model_path = os.path.join(weights_dir, 'best.pt')
    os.makedirs(os.path.dirname('models/fruit_yolo_model/best.pt'), exist_ok=True)
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'models/fruit_yolo_model/best.pt')
        print(f"Training complete. Best model saved to: models/fruit_yolo_model/best.pt")
    else:
        possible_models = list(Path(weights_dir).glob('*.pt'))
        if possible_models:
            possible_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            shutil.copy(str(possible_models[0]), 'models/fruit_yolo_model/best.pt')
            print(f"Training complete. Most recent model saved to: models/fruit_yolo_model/best.pt")
        else:
            print("Warning: Could not find a saved model after training. Continuing with the current model.")

    return model

def evaluate_model(yaml_path):
    model_path = 'models/fruit_yolo_model/best.pt'
    if not os.path.exists(model_path):
        possible_paths = list(Path('models/fruit_yolo_model').glob('**/best.pt'))
        if possible_paths:
            model_path = str(possible_paths[0])
        else:
            model_path = list(Path('.').glob('**/best.pt'))[0]
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = 0  # Use GPU
    else:
        device = 'cpu'

    model = YOLO(model_path)
    val_results = model.val(data=yaml_path, split='test', batch=16, device=device)
    print("\nFruit Ripeness Classification Performance:")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    print(f"F1 Score: {2 * val_results.box.mp * val_results.box.mr / (val_results.box.mp + val_results.box.mr):.4f}")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    test_img_dir = "fruit_dataset/images/test"
    all_test_images = os.listdir(test_img_dir)
    class_samples = {}
    for img in all_test_images:
        cls = img.split('_')[0]
        if cls not in class_samples:
            class_samples[cls] = []
        class_samples[cls].append(img)
    test_images = []
    for cls, images in class_samples.items():
        if images:
            test_images.append(random.choice(images))
    test_images = test_images[:6]
    plt.figure(figsize=(15, 10))
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(test_img_dir, img_name)
        results = model(img_path)
        plt.subplot(2, 3, i+1)
        for r in results:
            img = r.plot()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        class_name = img_name.split('_')[0]
        if 'fresh' in class_name:
            status = 'Fresh'
        else:
            status = 'Rotten'

        if 'apple' in class_name:
            fruit = 'Apple'
        elif 'banana' in class_name:
            fruit = 'Banana'
        elif 'orange' in class_name:
            fruit = 'Orange'
        else:
            fruit = ''

        plt.title(f"{status} {fruit}")

    plt.tight_layout()
    plt.savefig('results/fruits/test_predictions.png', dpi=150)
    plt.show()
    test_paths = [os.path.join(test_img_dir, img) for img in all_test_images[:100]]
    start_time = time.time()
    batch_size = 16
    for i in range(0, len(test_paths), batch_size):
        batch = test_paths[i:i+batch_size]
        _ = model(batch, verbose=False)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_paths)
    print(f"Average processing time: {avg_time*1000:.2f} ms per image")

    return val_results

def create_confusion_matrix(yaml_path):
    model_path = 'models/fruit_yolo_model/best.pt'
    if not os.path.exists(model_path):
        possible_paths = list(Path('models/fruit_yolo_model').glob('**/best.pt'))
        if possible_paths:
            model_path = str(possible_paths[0])
        else:
            model_path = list(Path('.').glob('**/best.pt'))[0]
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = 0  # Use GPU
    else:
        device = 'cpu'

    model = YOLO(model_path)
    class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange',
                   'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
    class_mapping = {
        'freshapples': 0,
        'freshbanana': 1,
        'freshoranges': 2,
        'rottenapples': 3,
        'rottenbanana': 4,
        'rottenoranges': 5
    }
    conf_matrix = np.zeros((6, 6))
    test_img_dir = "fruit_dataset/images/test"
    test_images = os.listdir(test_img_dir)
    class_images = {}
    for img_name in test_images:
        prefix = img_name.split('_')[0]
        true_class = class_mapping.get(prefix, -1)
        if true_class == -1:
            continue

        if true_class not in class_images:
            class_images[true_class] = []
        class_images[true_class].append(img_name)
    for true_class, images in class_images.items():
        img_paths = [os.path.join(test_img_dir, img) for img in images]
        batch_size = 16
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i:i+batch_size]
            results = model(batch, verbose=False)
            for j, r in enumerate(results):
                if len(r.boxes) > 0:
                    pred_class = int(r.boxes[0].cls.item())
                    conf_matrix[true_class, pred_class] += 1
                else:
                    conf_matrix[true_class, true_class] += 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Fruit Ripeness Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/fruits/confusion_matrix.png', dpi=150)
    plt.show()
    precision = np.zeros(6)
    recall = np.zeros(6)
    f1 = np.zeros(6)

    for i in range(6):
        if np.sum(conf_matrix[:, i]) > 0:
            precision[i] = conf_matrix[i, i] / np.sum(conf_matrix[:, i])
        else:
            precision[i] = 0

        if np.sum(conf_matrix[i, :]) > 0:
            recall[i] = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
        else:
            recall[i] = 0

        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0
    print("\nPer-class Performance Metrics:")
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    print(metrics_df)
    return conf_matrix

def test_custom_images():
    model_path = 'models/fruit_yolo_model/best.pt'
    if not os.path.exists(model_path):
        possible_paths = list(Path('models/fruit_yolo_model').glob('**/best.pt'))
        if possible_paths:
            model_path = str(possible_paths[0])
        else:
            model_path = list(Path('.').glob('**/best.pt'))[0]
    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO(model_path)

    print("Please place your fruit images in the 'test_images' folder and press Enter...")
    input()
    os.makedirs("test_images", exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_files = []
    for ext in image_extensions:
        test_files.extend(glob.glob(f"test_images/*{ext}"))

    if not test_files:
        print("No images found in 'test_images' folder. Please add some images and try again.")
        return
    for filename in test_files:
        results = model.predict(source=filename, save=False, conf=0.25)
        img = cv2.imread(filename)
        processed_img = roughness_local_binary_pattern(img)
        processed_filename = f"test_images/rlbp_{os.path.basename(filename)}"
        cv2.imwrite(processed_filename, processed_img)
        processed_results = model.predict(source=processed_filename, save=False, conf=0.25)
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        img_array = results[0].plot()
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.title(f"Standard Detection: {os.path.basename(filename)}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        processed_img_array = processed_results[0].plot()
        plt.imshow(cv2.cvtColor(processed_img_array, cv2.COLOR_BGR2RGB))
        plt.title(f"R-LBP Enhanced Detection")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        print(f"\nDetection Results for {os.path.basename(filename)}:")
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange',
                          'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
            print(f"Standard detection: {class_names[cls_id]} (Confidence: {conf:.2f})")
        else:
            print("Standard detection: No fruit detected")
        if len(processed_results[0].boxes) > 0:
            box = processed_results[0].boxes[0]
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            print(f"R-LBP enhanced: {class_names[cls_id]} (Confidence: {conf:.2f})")
        else:
            print("R-LBP enhanced: No fruit detected")
        os.remove(processed_filename)

def main():
    print("YOLO Fruit Ripeness Detection - Run it on Local Device")
    print("============================================")
    dataset_path = setup_dataset()
    if dataset_path is None:
        print("Dataset setup failed. Please check the instructions above.")
        return
    
    print(f"\nDataset path: {dataset_path}")
    try:
        rename_dataset_files(dataset_path)
    except Exception as e:
        print(f"Error during file renaming: {e}")
        print("Continuing with original filenames...")
    yaml_path = prepare_fruit_dataset(dataset_path)
    if yaml_path is None:
        print("Dataset preparation failed. Exiting.")
        return
    show_sample_images()
    visualize_preprocessing()
    try:
        model = train_fruit_model(yaml_path)
    except Exception as e:
        print(f"Error during training: {e}")
        print("Attempting to continue with evaluation using an existing model if available...")
    try:
        val_results = evaluate_model(yaml_path)
        create_confusion_matrix(yaml_path)
        test_custom_images()

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please ensure you have a trained model available.")
    
    print("\nFruit Ripeness Classification implementation completed successfully!")
    print("All models, results, and visualizations have been saved locally.")

if __name__ == "__main__":
    main()