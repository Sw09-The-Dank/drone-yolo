"""Main entry point - does nothing, allows manual command execution"""


def main():
    """Main function that does nothing - container stays running for manual commands"""
    print("Container is ready.")
    print("Commands:")
    print("  docker-compose exec yolo python create_data.py")
    print("  docker-compose exec yolo python train.py")
    print("  docker-compose exec yolo python inference.py --filename--")
    print("  docker-compose exec yolo python clear_data.py")
    
    # Keep container running indefinitely
    import time
    try:
        while True:
            time.sleep(3600)  # Sleep in 1-hour increments (keeps container alive)
    except KeyboardInterrupt:
        print("\nShutting down...")  


if __name__ == "__main__":
    main()

