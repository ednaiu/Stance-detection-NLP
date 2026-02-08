#!/bin/bash

# Deploy stance detection as WSGI application on reg.ru shared hosting
# This script sets up the application to run without Docker

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   STANCE DETECTION - WSGI DEPLOYMENT ON SHARED HOSTING         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

USER=${1:-u3089870}
HOST=${2:-31.31.198.9}
APP_DIR="/var/www/$USER/data/stance-detection"
VENV_DIR="$APP_DIR/venv"

echo "📋 Deployment Configuration:"
echo "  User: $USER"
echo "  Host: $HOST"
echo "  App Directory: $APP_DIR"
echo "  Virtual Environment: $VENV_DIR"
echo ""

# Step 1: Create app directory
echo "Step 1: Creating application directory..."
ssh $USER@$HOST "mkdir -p $APP_DIR && chmod 755 $APP_DIR" 2>/dev/null || echo "Directory may already exist"
echo "✓ Directory created"
echo ""

# Step 2: Clone or pull repository
echo "Step 2: Setting up application files..."
ssh $USER@$HOST << 'EOF'
cd /var/www/$USER/data/stance-detection

# Initialize git if needed
if [ ! -d .git ]; then
    echo "Cloning repository..."
    git clone https://github.com/ednaiu/Stance-detection.git .
else
    echo "Pulling latest changes..."
    git pull origin main
fi

# Create data and models directories
mkdir -p data/processed data/raw models
echo "✓ Directories created"
EOF
echo "✓ Application files prepared"
echo ""

# Step 3: Setup virtual environment
echo "Step 3: Setting up Python virtual environment..."
ssh $USER@$HOST << 'EOF'
cd /var/www/$USER/data/stance-detection

# Check if venv exists
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate venv and install dependencies
source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r StanceClassifier/requirements.txt 2>&1 | tail -5

echo "✓ Virtual environment ready"
EOF
echo "✓ Python environment prepared"
echo ""

# Step 4: Download pretrained models (optional, large ~2GB)
echo "Step 4: Checking models..."
ssh $USER@$HOST << 'EOF'
cd /var/www/$USER/data/stance-detection

if [ ! -f "models/sentence_embedding_baseline/pytorch_model.bin" ]; then
    echo "⚠ Pre-trained model not found."
    echo "Models can be:"
    echo "  1. Downloaded automatically on first request (slow)"
    echo "  2. Downloaded locally and uploaded via SCP"
    echo "  3. Trained locally first"
    echo ""
    echo "For now, create placeholder..."
    mkdir -p models/sentence_embedding_baseline
    touch models/sentence_embedding_baseline/.placeholder
else
    echo "✓ Models found"
fi
EOF
echo "✓ Models checked"
echo ""

# Step 5: Create startup script
echo "Step 5: Creating startup script..."
ssh $USER@$HOST << 'EOF'
cat > /var/www/$USER/data/stance-detection/run.sh << 'RUNSCRIPT'
#!/bin/bash
cd /var/www/$USER/data/stance-detection
source venv/bin/activate
export PYTHONUNBUFFERED=1
export MODEL_PATH=models/sentence_embedding_baseline
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 60 --access-logfile - wsgi:app
RUNSCRIPT

chmod +x /var/www/$USER/data/stance-detection/run.sh
echo "✓ Startup script created"
EOF
echo "✓ Startup script ready"
echo ""

# Step 6: Create systemd service file (if sudo available)
echo "Step 6: Checking systemd service setup..."
ssh $USER@$HOST << 'EOF'
if command -v systemctl &> /dev/null; then
    echo "⚠ Systemd available, but cannot create service without sudo"
    echo "  Contact hosting support to run the service"
else
    echo "Using alternative startup method"
fi
EOF
echo ""

# Step 7: Display next steps
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    DEPLOYMENT COMPLETE                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1️⃣ Test the application manually:"
echo "   ssh $USER@$HOST"
echo "   cd /var/www/$USER/data/stance-detection"
echo "   source venv/bin/activate"
echo "   ./run.sh"
echo ""
echo "2️⃣ Then open in browser:"
echo "   http://31.31.198.9:5000/health"
echo ""
echo "3️⃣ For permanent running, contact reg.ru support to:"
echo "   - Run the application as a background service"
echo "   - Or set up a cron job to keep it alive"
echo ""
echo "4️⃣ Or use screen/tmux for manual background running:"
echo "   ssh $USER@$HOST"
echo "   cd /var/www/$USER/data/stance-detection"
echo "   screen -S stance"
echo "   source venv/bin/activate"
echo "   ./run.sh"
echo "   # Press Ctrl+A then D to detach"
echo ""
echo "📚 API Endpoints:"
echo "   GET  http://31.31.198.9:5000/health"
echo "   GET  http://31.31.198.9:5000/"
echo "   POST http://31.31.198.9:5000/classify"
echo "   POST http://31.31.198.9:5000/classify_batch"
echo ""
