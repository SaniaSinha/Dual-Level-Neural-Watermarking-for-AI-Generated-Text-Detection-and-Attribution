from config import SOURCE_TABLE
from dual_system import DualLevelWatermarkingSystem


def main():
    system = DualLevelWatermarkingSystem()

    print("1. AI Detection Only")
    print("2. Encode Watermark")
    print("3. Decode Watermarked Text")

    choice = input("Choose option: ")

    if choice == "1":
        text = input("Enter sentence: ")
        result = system.detect_sentence(text)

        for key, value in result.items():
            print(f"{key}: {value}")

    elif choice == "2":
        text = input("Enter AI-generated text: ")
        text_id = int(input("Enter text ID: "))

        print("\nAvailable Source IDs:")
        for source_id, source_name in SOURCE_TABLE.items():
            print(f"{source_id}: {source_name}")

        source_id = input("Enter source ID to embed: ")

        result = system.encode(
            text=text,
            text_id=text_id,
            source_id=source_id
        )

        for key, value in result.items():
            print(f"{key}: {value}")

    elif choice == "3":
        text = input("Paste watermarked text: ")
        text_id = int(input("Enter text ID used during encoding: "))

        result = system.decode(
            watermarked_text=text,
            text_id=text_id
        )

        for key, value in result.items():
            print(f"{key}: {value}")

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
