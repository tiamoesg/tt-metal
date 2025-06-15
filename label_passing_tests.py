import re
from pathlib import Path

RESULT_FILE = Path("build_Release/results.txt")
CMAKE_FILE = Path("tt-train/tests/CMakeLists.txt")
LABEL_NAME = "CI_GREEN"

# 1. Parse results.txt for passed test names
def extract_passing_tests():
    passed_tests = []
    with RESULT_FILE.open("r") as f:
        for line in f:
            match = re.search(r"Test #[0-9]+: ([^\s]+)\s+\.+\s+Passed", line)
            if match:
                passed_tests.append(match.group(1))
    return passed_tests

# 2. Create the label block
def generate_label_block(test_names):
    lines = ["set_tests_properties("]
    for name in test_names:
        lines.append(f"    {name}")
    lines.append(f"    PROPERTIES LABELS {LABEL_NAME}")
    lines.append(")\n")
    return "\n".join(lines)

# 3. Inject or replace CI_GREEN block in CMakeLists.txt
def update_cmakelists(block):
    content = CMAKE_FILE.read_text()

    pattern = r"set_tests_properties\([\s\S]*?PROPERTIES LABELS " + LABEL_NAME + r"[\s\S]*?\)\n"
    if re.search(pattern, content):
        new_content = re.sub(pattern, block, content)
    else:
        new_content = content + "\n\n" + block

    if new_content != content:
        print("[INFO] Updating CMakeLists.txt with new CI_GREEN test block...")
        CMAKE_FILE.write_text(new_content)
        return True
    else:
        print("[INFO] No changes to CMakeLists.txt")
        return False

if __name__ == "__main__":
    if not RESULT_FILE.exists():
        print(f"[ERROR] {RESULT_FILE} not found. Run tests and save output to that file.")
        exit(1)

    passing_tests = extract_passing_tests()
    if not passing_tests:
        print("[WARN] No passing tests found.")
        exit(0)

    label_block = generate_label_block(passing_tests)
    changed = update_cmakelists(label_block)

    if changed:
        print("[ACTION] Run the following to rebuild:")
        print("  cd build_Release && cmake .. && ninja")
