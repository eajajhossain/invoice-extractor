
#  Pickle compatibility classes 
class Vocabulary:
    pass

class LabelEncoder:
    pass

from src.models.ner_model import NERModel

class Vocabulary:
    """
    Dummy class to satisfy pickle loading.
    The actual data is stored in attributes.
    """
    def __init__(self):
        pass


from src.models.ner_model import NERModel

def main():
    model = NERModel()

    sample_text = """
    Invoice No: INV-2024-0156
    Vendor: ACME SUPPLIES
    Total: $587.40
    """

    entities = model.predict(sample_text)

    print("\n=== PREDICTED ENTITIES ===")
    for ent in entities:
        print(ent)

    if not entities:
        print("⚠️ No entities detected")

if __name__ == "__main__":
    main()

