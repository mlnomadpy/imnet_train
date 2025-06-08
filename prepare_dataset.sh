#!/bin/bash

# ImageNet Dataset Preparation for GCS (Cost-Effective)
# Only prepares data, does NOT create expensive GPU instances

set -e

PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-imagenet-data"}

echo "💾 ImageNet Dataset Preparation (Cost-Effective)"
echo "=============================================="
echo "✅ Download ImageNet from Kaggle"
echo "✅ Process and organize data"  
echo "✅ Upload to Google Cloud Storage"
echo "❌ NO expensive GPU instances created"
echo ""

# Check project
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No project ID. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
echo "✅ Project: $PROJECT_ID"

# Install Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip3 install --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials needed!"
    echo "1. Get kaggle.json from https://www.kaggle.com/account"
    echo "2. Upload to Cloud Shell or save to ~/.kaggle/"
    echo "3. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    read -p "Press Enter when ready..."
fi

# Enable APIs and create bucket
echo "🔧 Setting up GCS..."
gcloud services enable storage-api.googleapis.com --quiet
if ! gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    gsutil mb gs://$BUCKET_NAME
fi

# Check competition access
echo "🏆 Checking ImageNet competition access..."
if ! kaggle competitions list | grep -q "imagenet-object-localization-challenge"; then
    echo "⚠️  Join competition: https://www.kaggle.com/c/imagenet-object-localization-challenge"
    read -p "Press Enter after joining..."
fi

# Download and process
echo "📥 Downloading ImageNet (~150GB, 2-4 hours)..."
mkdir -p ~/imagenet_work && cd ~/imagenet_work

# Check disk space
space=$(df /home -BG 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "20")
echo "💽 Available space: ${space}GB"

if [ "$space" -lt 200 ]; then
    echo "🚀 Streaming approach (limited space)..."
    # Download and upload files one by one
    kaggle competitions files -c imagenet-object-localization-challenge | tail -n +2 | while read line; do
        file=$(echo "$line" | awk '{print $1}')
        if [[ "$file" == *.zip ]]; then
            echo "📦 Processing: $file"
            kaggle competitions download -c imagenet-object-localization-challenge -f "$file"
            gsutil cp "$file" gs://$BUCKET_NAME/raw/
            
            # Process main data files
            if [[ "$file" == *"Data"* ]]; then
                unzip -q "$file"
                python3 -c "
import os, shutil
paths = ['./ILSVRC/Data/CLS-LOC', './ILSVRC', './Data/CLS-LOC', './Data']
for p in paths:
    if os.path.exists(p) and os.path.exists(p+'/train') and os.path.exists(p+'/val'):
        print(f'Found ImageNet at: {p}')
        if p != './imagenet_organized':
            shutil.copytree(p, './imagenet_organized', dirs_exist_ok=True)
        break
"
                if [ -d "./imagenet_organized" ]; then
                    gsutil -m cp -r ./imagenet_organized gs://$BUCKET_NAME/processed/
                    rm -rf ./imagenet_organized ./ILSVRC ./Data
                fi
            fi
            rm -f "$file"
        fi
    done
else
    echo "🚀 Local processing approach..."
    kaggle competitions download -c imagenet-object-localization-challenge
    
    for zip in *.zip; do
        [ -f "$zip" ] && unzip -q "$zip"
    done
    
    python3 -c "
import os, shutil
paths = ['./ILSVRC/Data/CLS-LOC', './ILSVRC', './Data/CLS-LOC', './Data']
for p in paths:
    if os.path.exists(p) and os.path.exists(p+'/train') and os.path.exists(p+'/val'):
        print(f'✅ Found ImageNet at: {p}')
        train_classes = len([d for d in os.listdir(p+'/train') if os.path.isdir(os.path.join(p+'/train', d))])
        print(f'📊 Classes: {train_classes}')
        if p != './imagenet_organized':
            shutil.copytree(p, './imagenet_organized', dirs_exist_ok=True)
        break
"
    
    # Upload everything
    gsutil -m cp *.zip gs://$BUCKET_NAME/raw/
    if [ -d "./imagenet_organized" ]; then
        gsutil -m cp -r ./imagenet_organized gs://$BUCKET_NAME/processed/
    fi
fi

# Create info file
cat > dataset_info.txt << EOF
ImageNet Dataset Ready
=====================
Bucket: gs://$BUCKET_NAME
Processed: $(date)

Structure:
- gs://$BUCKET_NAME/raw/          # Original zips
- gs://$BUCKET_NAME/processed/    # Organized data

Usage for training:
gsutil -m cp -r gs://$BUCKET_NAME/processed/imagenet_organized ~/data/

Classes: 1000
Training: ~1.2M images  
Validation: ~50K images
EOF

gsutil cp dataset_info.txt gs://$BUCKET_NAME/

# Cleanup
cd ~ && rm -rf ~/imagenet_work

echo ""
echo "🎉 Dataset Ready!"
echo "📊 Bucket: gs://$BUCKET_NAME"
echo "💰 Cost: ~$5-10 (no GPU costs)"
echo ""
echo "🚀 To train later:"
echo "1. Create GPU instance when ready"
echo "2. Download: gsutil -m cp -r gs://$BUCKET_NAME/processed/imagenet_organized ~/data/"
echo "3. Run training"
echo "4. Stop instance when done!" 