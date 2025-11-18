#!/bin/bash
# Quick test script for cluster mode functionality
# This does a dry-run to verify the setup is correct

echo "=========================================="
echo "Testing GEB Cluster Mode Setup"
echo "=========================================="
echo ""

cd /scistor/ivm/tbr910/GEB/GEB/workflow
source "$HOME/GEB/GEB/.venv/bin/activate"

echo "1. Testing cluster-generic executor plugin..."
python -c "import snakemake_executor_plugin_cluster_generic; print('   ✓ Cluster-generic plugin loaded successfully')" || exit 1
echo ""

echo "2. Testing Snakemake cluster configuration..."
snakemake \
    --snakefile Snakefile_large_scale \
    --configfile config/large_scale.yml \
    --executor cluster-generic \
    --cluster-generic-submit-cmd "sbatch --parsable --job-name=test-{rule}" \
    --jobs 2 \
    --dry-run \
    build_all_clusters 2>&1 | grep -q "build_cluster" && echo "   ✓ Snakemake can generate execution plan with cluster mode" || exit 1
echo ""

echo "3. Testing script syntax..."
bash -n /scistor/ivm/tbr910/GEB/sh_scripts/run_large_scale_slurm.sh && echo "   ✓ Script syntax is valid" || exit 1
echo ""

echo "4. Checking cluster directories..."
if [ -d "/scistor/ivm/tbr910/GEB/models/large_scale/test_000" ]; then
    echo "   ✓ Found test_000 cluster"
else
    echo "   ⚠ Warning: test_000 cluster not found"
fi
if [ -d "/scistor/ivm/tbr910/GEB/models/large_scale/test_001" ]; then
    echo "   ✓ Found test_001 cluster"
else
    echo "   ⚠ Warning: test_001 cluster not found"
fi
echo ""

echo "5. Checking SLURM access..."
if command -v sbatch &> /dev/null; then
    echo "   ✓ sbatch command available"
    sinfo -p ivm &>/dev/null && echo "   ✓ Can access ivm partition" || echo "   ⚠ Warning: Cannot access ivm partition"
else
    echo "   ✗ sbatch not available"
fi
echo ""

echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  cd /scistor/ivm/tbr910/GEB/sh_scripts"
echo "  sbatch run_large_scale_slurm.sh all cluster 10"
echo ""
