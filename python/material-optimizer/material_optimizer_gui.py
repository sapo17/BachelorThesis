"""
Author: Can Hasbay
"""

from PyQt6.QtWidgets import QApplication
import sys
import logging
import ctypes
from src.constants import LOG_FILE, MY_APP_ID
from src.material_optimizer_model import MaterialOptimizerModel
from src.material_optimizer_view import MaterialOptimizerView
from src.material_optimizer_controller import MaterialOptimizerController


def main():
    # On windows: name the application
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            MY_APP_ID
        )

    # Prepare log file
    LOG_FILE.unlink(missing_ok=True)
    logging.basicConfig(
        filename=LOG_FILE, level=logging.INFO
    )

    # initialize the PyQt application
    app = QApplication(sys.argv)

    # Initialize the Model
    materialOptimizerModel = MaterialOptimizerModel()

    # Initialize the View
    materialOptimizerView = MaterialOptimizerView()
    materialOptimizerView.show()

    # Initialize the Controller
    materialOptimizerController = MaterialOptimizerController(
        model=materialOptimizerModel, view=materialOptimizerView
    )

    # execute the application
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing material-optmizer...')


if __name__ == "__main__":
    main()
