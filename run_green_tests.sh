#!/bin/bash

cd "$(dirname "$0")/build_Release"

echo "🧪 Running full test suite..."
ctest --output-on-failure | tee results.txt

cd ..

echo "🏷️ Labeling passing tests..."
python3 label_passing_tests.py

echo "🔧 Rebuilding project..."
cd build_Release
cmake .. && ninja

echo "✅ Running green tests..."
ctest -L CI_GREEN --output-on-failure
