"""Main entry point - does nothing, allows manual command execution"""


def main():
    """Main function that does nothing - container stays running for manual commands"""
    print("Container is ready.")
    print("  docker-compose exec yolo python clear_data.py")
    print("  docker-compose exec yolo python create_data.py")
    print("  docker-compose exec yolo python train.py")
    print("  docker-compose exec yolo python inference.py")
    
    # Keep container running
    import time
    while True:
        time.sleep(3600)  


if __name__ == "__main__":
    main()

