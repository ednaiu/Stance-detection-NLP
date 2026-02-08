#!/bin/bash

# Server readiness check script
# Usage: bash check_server.sh

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║             SERVER READINESS CHECK 31.31.198.9                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# Helper function for checks
check_command() {
    local name=$1
    local cmd=$2
    echo -n "Checking $name... "
    
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAIL++))
        return 1
    fi
}

echo -e "${BLUE}1. BASIC SYSTEM INFO${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo ""

echo -e "${BLUE}2. DOCKER INSTALLATION${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_command "Docker installed" "docker --version"
if [ $? -eq 0 ]; then
    docker --version | sed 's/^/  /'
fi

check_command "Docker daemon running" "docker ps"
if [ $? -eq 0 ]; then
    echo "  $(docker ps | wc -l) container(s) running"
fi

check_command "Docker user permissions" "docker run --rm hello-world"
echo ""

echo -e "${BLUE}3. REQUIRED DIRECTORIES${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_command "/data/stance-detection/data exists" "test -d /data/stance-detection/data"
check_command "/data/stance-detection/models exists" "test -d /data/stance-detection/models"

if [ -d /data/stance-detection ]; then
    echo "  Directory structure:"
    ls -la /data/stance-detection/ | sed 's/^/    /'
fi
echo ""

echo -e "${BLUE}4. DISK SPACE${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check root disk
root_free=$(df / | tail -1 | awk '{print $4}')
root_total=$(df / | tail -1 | awk '{print $2}')

if [ "$root_free" -gt 5242880 ]; then
    echo -e "${GREEN}✓ Root filesystem${NC}: $(numfmt --to=iec-i --suffix=B $root_free 2>/dev/null || echo "$root_free KB") free (sufficient)"
    ((PASS++))
else
    echo -e "${RED}✗ Root filesystem${NC}: $(numfmt --to=iec-i --suffix=B $root_free 2>/dev/null || echo "$root_free KB") free (need 5GB+)"
    ((FAIL++))
fi

# Check /data disk if it exists
if [ -d /data ]; then
    data_free=$(df /data | tail -1 | awk '{print $4}')
    if [ "$data_free" -gt 5242880 ]; then
        echo -e "${GREEN}✓ /data partition${NC}: $(numfmt --to=iec-i --suffix=B $data_free 2>/dev/null || echo "$data_free KB") free (sufficient)"
        ((PASS++))
    else
        echo -e "${RED}✗ /data partition${NC}: $(numfmt --to=iec-i --suffix=B $data_free 2>/dev/null || echo "$data_free KB") free (need 5GB+)"
        ((FAIL++))
    fi
fi
echo ""

echo -e "${BLUE}5. NETWORK & PORTS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Open ports (TCP):"
ss -tlnp 2>/dev/null | grep LISTEN | awk '{print "  " $4}' || netstat -tlnp 2>/dev/null | grep LISTEN | awk '{print "  " $4}' || echo "  (unable to check)"

# Check port 22
if ss -tlnp 2>/dev/null | grep -q ":22 "; then
    echo -e "${GREEN}✓ SSH (port 22)${NC}: OPEN"
    ((PASS++))
elif netstat -tlnp 2>/dev/null | grep -q ":22 "; then
    echo -e "${GREEN}✓ SSH (port 22)${NC}: OPEN"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ SSH (port 22)${NC}: Status unknown"
fi

# Check port 5000
if ss -tlnp 2>/dev/null | grep -q ":5000 "; then
    echo -e "${GREEN}✓ Flask API (port 5000)${NC}: OPEN"
    ((PASS++))
elif netstat -tlnp 2>/dev/null | grep -q ":5000 "; then
    echo -e "${GREEN}✓ Flask API (port 5000)${NC}: OPEN"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ Flask API (port 5000)${NC}: NOT OPEN (will be opened when container starts)"
fi
echo ""

echo -e "${BLUE}6. ENVIRONMENT VARIABLES${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "  HOME: $HOME"
echo "  USER: $USER"
echo "  PATH: $(echo $PATH | tr ':' '\n' | head -3 | sed 's/^/    /')"
echo ""

echo -e "${BLUE}7. SUMMARY${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "Checks passed: ${GREEN}$PASS${NC}"
echo -e "Checks failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        ✓ SERVER IS READY FOR DEPLOYMENT                         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ✗ PLEASE FIX ISSUES BEFORE DEPLOYING                          ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
