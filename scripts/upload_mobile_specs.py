#!/usr/bin/env python3
"""
Upload mobile phone specifications from CSV to Qdrant
"""

import os

os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"

import sys
import csv
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from app.core.config import settings
from app.core.vectorstore import VectorStoreManager


def process_mobile_csv():
    """Process mobiles.csv and upload to Qdrant"""
    print("📱 Processing mobile phone specifications...")

    try:
        # Read CSV file
        csv_file = settings.data_dir / "mobiles_dataset" / "mobiles.csv"

        if not csv_file.exists():
            print(f"❌ CSV file not found: {csv_file}")
            return False

        documents = []

        # Read CSV data
        with open(csv_file, 'r', encoding='utf-8') as file:
            # Read header
            first_line = file.readline().strip()
            print(f"CSV header: {first_line}")

            # Reset to beginning
            file.seek(0)
            reader = csv.reader(file)

            # Skip header if it exists
            header = next(reader, None)
            if header and header[0].lower() in ['brand', 'бренд']:
                print("Found header, skipping...")
            else:
                # If no header, reset and use first line as data
                file.seek(0)
                reader = csv.reader(file)

            row_count = 0
            for row_num, row in enumerate(reader, 1):
                if len(row) >= 10:  # Ensure we have all expected columns
                    try:
                        # Parse CSV columns
                        brand = row[0].strip()
                        model = row[1].strip()
                        weight = row[2].strip()
                        ram = row[3].strip()
                        front_camera = row[4].strip()
                        main_camera = row[5].strip()
                        processor = row[6].strip()
                        battery = row[7].strip()
                        screen = row[8].strip()
                        price = row[9].strip()
                        year = row[10].strip() if len(row) > 10 else "N/A"

                        # Create structured content
                        phone_info = {
                            "brand": brand,
                            "model": model,
                            "weight": weight,
                            "ram": ram,
                            "front_camera": front_camera,
                            "main_camera": main_camera,
                            "processor": processor,
                            "battery": battery,
                            "screen": screen,
                            "price": price,
                            "year": year
                        }

                        # Create readable content
                        content = f"""
{brand} {model}

Основные характеристики:
• Бренд: {brand}
• Модель: {model}
• Вес: {weight}
• Оперативная память: {ram}
• Фронтальная камера: {front_camera}
• Основная камера: {main_camera}
• Процессор: {processor}
• Батарея: {battery}
• Экран: {screen}
• Цена: {price}
• Год выпуска: {year}

JSON: {json.dumps(phone_info, ensure_ascii=False)}
                        """.strip()

                        # Create document
                        doc = Document(
                            page_content=content,
                            metadata={
                                "brand": brand,
                                "model": model,
                                "price": price,
                                "year": year,
                                "phone_id": f"{brand}_{model}".replace(" ", "_"),
                                "content_type": "mobile_spec"
                            }
                        )
                        documents.append(doc)
                        row_count += 1

                        if row_count <= 5:  # Show first 5 processed
                            print(f"✅ Processed: {brand} {model} - {price}")

                    except Exception as e:
                        print(f"⚠️ Error processing row {row_num}: {e}")
                        print(f"   Row data: {row}")

                else:
                    print(f"⚠️ Skipping row {row_num}: insufficient columns ({len(row)})")

        print(f"\n📊 Processed {len(documents)} phones from CSV")

        if not documents:
            print("❌ No documents processed!")
            return False

        # Upload to Qdrant
        print(f"\n📤 Uploading to Qdrant collection '{settings.specs_collection}'...")

        vector_manager = VectorStoreManager()

        # Clear existing collection
        try:
            vector_manager.client.delete_collection(settings.specs_collection)
            print("🗑️ Cleared existing collection")
        except:
            pass

        # Create new collection with documents
        QdrantVectorStore.from_documents(
            documents,
            vector_manager.embeddings,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.specs_collection,
        
        )

        print("✅ Mobile specifications uploaded successfully!")

        # Show summary by brand
        brands = {}
        for doc in documents:
            brand = doc.metadata.get('brand', 'Unknown')
            brands[brand] = brands.get(brand, 0) + 1

        print(f"\n📊 Upload Summary:")
        print(f"   Total phones: {len(documents)}")
        for brand, count in sorted(brands.items()):
            print(f"   {brand}: {count} models")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uploaded_data():
    """Test the uploaded data"""
    print("\n🧪 Testing uploaded data...")

    try:
        vm = VectorStoreManager()
        specs_retriever, _ = vm.get_retrievers()

        test_queries = [
            "iPhone 15 Pro Max",
            "iPhone 16 Pro Max",
            "Samsung Galaxy",
            "Xiaomi"
        ]

        for query in test_queries:
            print(f"\n📱 Testing: '{query}'")
            docs = specs_retriever.invoke(query)
            print(f"   Found: {len(docs)} documents")

            if docs and docs[0].page_content:
                preview = docs[0].page_content[:100].replace('\n', ' ')
                print(f"   Preview: {preview}...")

                # Look for iPhone 16 Pro Max specifically
                if "iPhone 16 Pro Max" in docs[0].page_content:
                    print(f"   💰 Found iPhone 16 Pro Max pricing!")
            else:
                print(f"   ⚠️ No content found")

    except Exception as e:
        print(f"❌ Test error: {e}")


def main():
    print("🚀 Mobile Specifications Upload")
    print("=" * 40)

    if process_mobile_csv():
        test_uploaded_data()
        print(f"\n✅ Setup completed!")
        print(f"🎯 Your agent should now access iPhone prices from the CSV!")
    else:
        print(f"\n❌ Upload failed!")


if __name__ == "__main__":
    main()