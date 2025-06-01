#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Backend/Development Environments
# This script sets up a Python virtual environment and installs dependencies
# primarily for Tier 2 (Facility Node) and Tier 3 (Cloud Node) Python components,
# as well as for the overall project development and simulation environment.
# Personal Edge Devices (PEDs) and Supervisor Hubs running native applications
# have separate build/setup processes (e.g., Android Studio, embedded toolchains).

echo "Setting up Sentinel Health Co-Pilot Python virtual environment and installing dependencies..."
echo "Targeting: Python backend services, DHO/Manager web dashboards, development/simulation."

# --- Configuration ---
# Read from a config file or use defaults
PROJECT_ROOT_DIR=$(pwd) # Assume script is run from project root, or adjust as needed
VENV_NAME_PY="${VENV_NAME_PY:-venv_sentinel_py_backend}" # Allow override via ENV var, default name
VENV_DIR="${PROJECT_ROOT_DIR}/${VENV_NAME_PY}"
REQUIREMENTS_FILE="${PROJECT_ROOT_DIR}/requirements_backend.txt" # Specific requirements for backend/web
PYTHON_CMD="${PYTHON_CMD:-python3}" # Allow override, default to python3

# --- Helper Functions ---
log_info() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S'): $1"
}
log_warn() {
    echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S'): $1"
}
log_error() {
    echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S'): $1" >&2 # Errors to stderr
}
exit_on_error() {
    if [ $? -ne 0 ]; then
        log_error "$1"
        log_error "Setup aborted due to a critical error."
        # Attempt to deactivate venv if active, though it might not be set if creation failed
        if [ -n "$VIRTUAL_ENV" ]; then
            log_info "Attempting to deactivate virtual environment..."
            deactivate
        fi
        exit 1
    fi
}

# --- Pre-requisite Checks ---
log_info "Checking for ${PYTHON_CMD}..."
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    log_error "${PYTHON_CMD} command could not be found. Please ensure Python 3 (recommended 3.8+) is installed and in your PATH."
    exit 1
fi
${PYTHON_CMD} -c "import sys; exit(0) if sys.version_info >= (3, 8) else exit(1)"
if [ $? -ne 0 ]; then
    log_warn "Python version is older than 3.8. Some dependencies might require a newer version. Proceeding with caution."
fi

log_info "Checking for 'venv' module..."
if ! ${PYTHON_CMD} -m venv -h &> /dev/null; then # Simple way to check if module exists
    log_error "'venv' module not found for ${PYTHON_CMD}. This is typically part of a standard Python installation. Please check your Python setup (e.g., ensure 'python3-venv' package is installed on Debian/Ubuntu)."
    exit 1
fi

# --- Virtual Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating Python virtual environment in ./${VENV_NAME_PY}..."
    ${PYTHON_CMD} -m venv ${VENV_DIR}
    exit_on_error "Failed to create virtual environment. Check permissions and Python installation."
    log_info "Virtual environment created successfully."
else
    log_info "Python virtual environment ./${VENV_NAME_PY} already exists. Skipping creation."
fi

# --- Activate Virtual Environment ---
log_info "Activating Python virtual environment..."
# shellcheck source=./venv_sentinel_py_backend/bin/activate # Suppress ShellCheck SC1090/SC1091 for dynamic source
source "${VENV_DIR}/bin/activate"
# Robust check for activation (VIRTUAL_ENV variable should be set by activate script)
if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "${VENV_DIR}" ]; then
    log_error "Virtual environment activation failed or activated the wrong environment."
    log_error "Expected: ${VENV_DIR}, Actual: ${VIRTUAL_ENV:-'Not set'}"
    log_error "Please try activating manually: source ${VENV_DIR}/bin/activate"
    exit 1
fi
log_info "Virtual environment activated: $VIRTUAL_ENV"

# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip within the virtual environment..."
pip install --upgrade pip
exit_on_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE" ]; then
    log_info "Installing dependencies from ${REQUIREMENTS_FILE}..."
    # Consider adding flags like --no-cache-dir for CI environments
    # For Geopandas and its C dependencies (GDAL, GEOS, PROJ), installation can be complex.
    # Users might need to install these system libraries first (e.g., via apt, yum, brew, or conda).
    # The requirements_backend.txt should pin versions for reproducibility.
    pip install -r "${REQUIREMENTS_FILE}"
    if [ $? -eq 0 ]; then
        log_info "Python dependencies installed successfully."
    else
        log_warn "Failed to install some dependencies from ${REQUIREMENTS_FILE}."
        log_warn "This might be due to missing system libraries (e.g., for Geopandas like GDAL, GEOS, PROJ) or conflicts."
        log_warn "Please review the errors above. You may need to install system-level C libraries first."
        log_warn "Consult the project documentation for detailed setup instructions for complex packages."
        # Do not exit immediately; some parts of the backend might still work. Or exit if all critical.
        # exit_on_error "Dependency installation failed critically."
    fi
else
    log_warn "${REQUIREMENTS_FILE} not found. Skipping Python dependency installation."
    log_warn "Please ensure you have a ${REQUIREMENTS_FILE} with necessary packages (e.g., streamlit, pandas, geopandas, plotly, numpy, Flask/FastAPI, scikit-learn, tensorflow, etc.)."
fi

# --- Post-Setup Information & Next Steps ---
echo ""
log_info "Python backend/development environment setup complete for ${app_config.APP_NAME}."
log_info "Key paths:"
log_info "  Project Root: ${PROJECT_ROOT_DIR}"
log_info "  Virtual Env:  ${VENV_DIR}"
log_info "  Requirements: ${REQUIREMENTS_FILE}"
echo ""
log_info "To run Python-based components (e.g., Streamlit DHO dashboard simulation, Facility Node backend):"
log_info "1. Ensure the virtual environment is active: source ${VENV_DIR}/bin/activate"
log_info "2. Navigate to the appropriate subdirectory if needed."
log_info "3. Execute the Python script (e.g., 'streamlit run test/app_home.py' from project root, or specific backend service script)."
echo ""
log_info "For Personal Edge Device (PED) and native Supervisor Hub applications:"
log_info "- These require separate build environments (e.g., Android Studio for Android apps)."
log_info "- Edge AI models (.tflite) need to be compiled/converted and bundled with the native apps."
log_info "- Refer to the specific documentation for PED/Hub native application setup."
echo ""
log_info "To deactivate this Python virtual environment later, simply type: deactivate"
echo ""

# Example: Check if .env file for environment variables exists (e.g., MAPBOX_ACCESS_TOKEN)
ENV_FILE_EXAMPLE="${PROJECT_ROOT_DIR}/.env.example"
ENV_FILE_ACTUAL="${PROJECT_ROOT_DIR}/.env"
if [ -f "$ENV_FILE_EXAMPLE" ] && [ ! -f "$ENV_FILE_ACTUAL" ]; then
    log_warn "An example environment file '${ENV_FILE_EXAMPLE##*/}' exists."
    log_warn "Please copy it to '${ENV_FILE_ACTUAL##*/}' and fill in necessary environment variables (e.g., MAPBOX_ACCESS_TOKEN, database credentials for Facility Node)."
fi

exit 0
