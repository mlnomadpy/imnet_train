#!/bin/bash

# Create GPU Training Instance and Setup Environment
# Use this AFTER you've prepared the dataset with prepare_dataset.sh

set -e

PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-imagenet-data"}
ZONE=${ZONE:-"us-central1-a"}
INSTANCE_NAME=${INSTANCE_NAME:-"imagenet-training"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-8"}
GPU_TYPE=${GPU_TYPE:-"nvidia-tesla-v100"}

echo "🖥️  GPU Training Instance Setup"
echo "=============================="
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "GPU: $GPU_TYPE"
echo "Dataset: gs://$BUCKET_NAME"
echo ""

# Check if dataset exists
if ! gsutil ls gs://$BUCKET_NAME/processed/imagenet_organized/ &>/dev/null; then
    echo "❌ Dataset not found! Run prepare_dataset.sh first."
    echo "Expected: gs://$BUCKET_NAME/processed/imagenet_organized/"
    exit 1
fi
echo "✅ Dataset found in bucket"

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo "⚠️  Instance $INSTANCE_NAME already exists"
    echo "1. Delete it: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
    echo "2. Or use existing: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    exit 1
fi

echo "💰 Cost estimate: ~$2.50/hour for GPU instance"
read -p "Create GPU instance? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Create GPU instance
echo "🚀 Creating GPU instance..."
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

# Copy training files
echo "📂 Copying training files..."
gcloud compute scp train.py $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute scp requirements.txt $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute scp run_training.sh $INSTANCE_NAME:~/ --zone=$ZONE

# Create setup script for the instance
cat > setup_training_env.sh << EOF
#!/bin/bash
set -e

echo "🔧 Setting up training environment..."

# Install dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test GPU
echo "🔍 Testing GPU..."
nvidia-smi

# Create directories
mkdir -p ~/models ~/data

# Download dataset from GCS
echo "📥 Downloading ImageNet dataset from GCS..."
echo "This will take 10-20 minutes..."
gsutil -m cp -r gs://$BUCKET_NAME/processed/imagenet_organized ~/data/

# Verify dataset
if [ -d "~/data/imagenet_organized" ]; then
    echo "✅ Dataset downloaded successfully"
    echo "📊 Dataset info:"
    ls -la ~/data/imagenet_organized/
    echo "Classes: \$(ls ~/data/imagenet_organized/train/ | wc -l)"
else
    echo "❌ Dataset download failed"
    exit 1
fi

# Update train.py to use local data
sed -i 's|~/data/tfds_data|~/data/imagenet_organized|g' ~/train.py

# Make scripts executable
chmod +x ~/run_training.sh

echo ""
echo "✅ Setup complete!"
echo "🚀 To start training: ./run_training.sh"
echo "📊 Monitor: tail -f ~/training.log"
echo "💰 REMEMBER: Stop instance when done!"
EOF

# Copy and run setup script
echo "🔧 Setting up training environment on instance..."
gcloud compute scp setup_training_env.sh $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="chmod +x ~/setup_training_env.sh && ~/setup_training_env.sh"

echo ""
echo "🎉 Training Instance Ready!"
echo ""
echo "📍 Next Steps:"
echo "1. SSH to instance:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "2. Start training:"
echo "   ./run_training.sh"
echo ""
echo "3. Monitor from local machine:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f ~/training.log'"
echo ""
echo "4. Check GPU usage:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='nvidia-smi'"
echo ""
echo "5. ⚠️  IMPORTANT - Stop instance when done:"
echo "   gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "💰 Current cost: ~$2.50/hour while running"
echo "📊 Training time: ~24-48 hours for full ImageNet" 