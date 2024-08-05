class KVOverrideEntry(QWidget):
    """
    KVOverrideEntry is a PyQt6-based widget for creating and managing key-value override entries.

    This class provides functionality for:
    - Inputting keys and values with type specification
    - Dynamic value substitution using predefined placeholders
    - Validating inputs based on selected data types
    - Generating formatted override strings

    The widget includes input fields for keys and values, a type selector,
    and a delete button. It supports various system-related and custom placeholders
    for dynamic value generation.

    Attributes:
        deleted (pyqtSignal): Signal emitted when the entry is deleted.
        key_input (QLineEdit): Input field for the key.
        type_combo (QComboBox): Dropdown for selecting the value type.
        value_input (QLineEdit): Input field for the value.

    Supported dynamic placeholders:
        {system.time.milliseconds}: Current time in milliseconds
        {system.time.seconds}: Current time in seconds
        {system.date.iso}: Current date in ISO format
        {system.datetime.iso}: Current date and time in ISO format
        {system.username}: Current system username
        {system.hostname}: Current system hostname
        {system.platform}: Current operating system platform
        {system.python.version}: Python version
        {system.date}: Current date in YYYY-MM-DD format
        {model.name}: Model name (if provided)
        {quant.type}: Quantization type (if provided)
        {output.path}: Output path (if provided)
    """

    def __init__(self, parent=None):
        """
        Initialize the KVOverrideEntry widget.

        This method sets up the widget layout, creates and configures input fields,
        sets up validators, and connects signals to their respective slots.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """

    def delete_clicked(self):
        """
        Handle the delete button click event.

        Emits the 'deleted' signal to notify the parent widget that this entry
        should be removed.
        """

    def get_override_string(self, model_name=None, quant_type=None, output_path=None):
        """
        Generate a formatted override string with dynamic value substitution.

        This method processes the input fields and replaces any placeholders
        in the value with their corresponding dynamic values.

        Args:
            model_name (str, optional): Model name for substitution.
            quant_type (str, optional): Quantization type for substitution.
            output_path (str, optional): Output path for substitution.

        Returns:
            str: Formatted override string in the format "key=type:value".
        """

    def get_raw_override_string(self):
        """
        Generate a raw override string without dynamic substitution.

        Returns:
            str: Raw override string with placeholders intact, in the format "key=type:value".
        """

    def update_validator(self, type_):
        """
        Update the validator for the value input field based on the selected type.

        This method ensures that the value input adheres to the chosen data type.

        Args:
            type_ (str): The selected data type ('int', 'float', or 'str').
        """
