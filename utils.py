def continue_loop(message: str) -> bool:

    """
    Creates a continue [y/n] loop with a customized message

    Parameters
    -----------
     - `message`: message to be printed in the input
        function

    Returns
    -----------
     - `flag`: boolean value indicating whether the user
        wants to continue or not
    """

    while True:
        user_input = input(f"{message} [y\\n]: ")
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid input.")