import argparse
from signature_api.database import validate_pin

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Validate your PIN to access the SignaSure.")
    parser.add_argument('pin', type=str, help="The PIN provided to access the API.")
    args = parser.parse_args()

    # Validate the PIN
    if validate_pin(args.pin):
        print("Valid PIN. Welcome to SignaSure!")
    else:
        print("Invalid PIN. Access denied.")

if __name__ == "__main__":
    main()