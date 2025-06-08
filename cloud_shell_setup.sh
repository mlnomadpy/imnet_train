#!/bin/bash

# ==============================================================================
# ImageNet Training Setup Script for Google Cloud Shell
# ==============================================================================

set -e

PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-"us-central1-a"}
INSTANCE_NAME=${INSTANCE_NAME:-"imagenet-training-instance"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-8"}
GPU_TYPE=${GPU_TYPE:-"nvidia-tesla-a100"}
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-imagenet-data"}

echo "🚀 ImageNet Training Setup for Google Cloud Shell"
echo "================================================="

# Check Cloud Shell environment
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No project ID found. Please set one:"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✅ Using project: $PROJECT_ID"

# Install Kaggle CLI if needed
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip3 install --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token' to download kaggle.json"
    echo "3. In Cloud Shell, click the 'Upload file' button (⋮ menu)"
    echo "4. Upload your kaggle.json file"
    echo "5. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Press Enter when you've completed the Kaggle setup..."
fi

# Enable APIs
echo "🔧 Enabling GCP APIs..."
gcloud services enable compute.googleapis.com --quiet
gcloud services enable storage-api.googleapis.com --quiet

# Create bucket
echo "🪣 Creating GCS bucket: $BUCKET_NAME"
if ! gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    gsutil mb gs://$BUCKET_NAME
fi

# Check competition access
echo "🏆 Checking ImageNet competition access..."
if ! kaggle competitions list | grep -q "imagenet-object-localization-challenge"; then
    echo "⚠️  You need to join the ImageNet competition first:"
    echo "   https://www.kaggle.com/c/imagenet-object-localization-challenge"
    echo "   Click 'Join Competition' and accept the rules"
    read -p "Press Enter after joining the competition..."
fi

# Download ImageNet (streaming to GCS due to Cloud Shell space limits)
echo "📥 Downloading ImageNet dataset (streaming to GCS)..."
mkdir -p ~/temp_download
cd ~/temp_download

# Download files one by one to save space
kaggle competitions files -c imagenet-object-localization-challenge | tail -n +2 | while read -r line; do
    filename=$(echo "$line" | awk '{print $1}')
    if [[ "$filename" == *.zip ]]; then
        echo "Downloading: $filename"
        kaggle competitions download -c imagenet-object-localization-challenge -f "$filename"
        gsutil cp "$filename" gs://$BUCKET_NAME/raw/
        rm -f "$filename"
    fi
done

cd ~
rm -rf ~/temp_download

# Create GPU instance
echo "🖥️  Creating GPU instance: $INSTANCE_NAME"
if ! gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE 2>/dev/null; then
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator="type=$GPU_TYPE,count=1" \
        --boot-disk-size=200GB \
        --boot-disk-type=pd-ssd \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True"
    
    echo "⏱️  Waiting for instance to be ready..."
    sleep 60
fi

# Copy files to instance
echo "📂 Copying files to instance..."
gcloud compute scp train.py $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute scp requirements.txt $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute scp run_training.sh $INSTANCE_NAME:~/ --zone=$ZONE

# Create setup script for instance
cat > ~/setup_instance.sh << EOF
#!/bin/bash
set -e

echo "🔧 Setting up training environment..."

# Install dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Create directories
mkdir -p ~/models ~/data

# Download and process ImageNet data
echo "📥 Downloading ImageNet data from GCS..."
gsutil -m cp gs://$BUCKET_NAME/raw/*.zip ~/data/

echo "📦 Extracting ImageNet data..."
cd ~/data
for zip in *.zip; do
    if [ -f "\$zip" ]; then
        unzip -q "\$zip"
    fi
done

# Organize data structure
python3 << 'PYTHON_EOF'
import os
import shutil

# Find ImageNet structure
paths = ["./ILSVRC/Data/CLS-LOC", "./ILSVRC", "./Data/CLS-LOC", "./Data"]
for path in paths:
    if os.path.exists(path):
        train_path = os.path.join(path, "train")
        val_path = os.path.join(path, "val")
        if os.path.exists(train_path) and os.path.exists(val_path):
            if path != "./imagenet_data":
                shutil.copytree(path, "./imagenet_data")
            print(f"✅ Found ImageNet at: {path}")
            break
PYTHON_EOF

# Create symlinks for train.py
if [ -d "./imagenet_data" ]; then
    ln -sf ~/data/imagenet_data ~/data/raw_imagenet_data
fi

# Cleanup zip files to save space
rm -f *.zip

chmod +x ~/run_training.sh
echo "✅ Setup complete! Run: ./run_training.sh"
EOF

# Copy and run setup script on instance
gcloud compute scp ~/setup_instance.sh $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="chmod +x ~/setup_instance.sh && ~/setup_instance.sh"

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "📍 Next Steps:"
echo "   1. SSH to your instance:"
echo "      gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "   2. Start training:"
echo "      ./run_training.sh"
echo ""
echo "   3. Monitor from Cloud Shell:"
echo "      gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f ~/training.log'"
echo ""
echo "   4. Stop instance when done:"
echo "      gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "💰 Remember: GPU instances cost ~$2.50/hour!" 