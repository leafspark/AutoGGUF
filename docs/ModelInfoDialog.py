class ModelInfoDialog(QDialog):
    """
    A dialog window for displaying model information.

    This class creates a dialog that shows detailed information about a machine learning model,
    including its architecture, quantization type, and other relevant data.

    Attributes:
        None

    Args:
        model_info (dict): A dictionary containing the model's information.
        parent (QWidget, optional): The parent widget of this dialog. Defaults to None.
    """

    def format_model_info(self, model_info) -> str:
        """
        Formats the model information into HTML for display.

        This method takes the raw model information and converts it into a formatted HTML string,
        which can be displayed in the dialog's QTextEdit widget.

        Args:
            model_info (dict): A dictionary containing the model's information.

        Returns:
            str: Formatted HTML string containing the model information.
        """
