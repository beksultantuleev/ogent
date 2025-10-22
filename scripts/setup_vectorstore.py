"""
Setup script for vector store data upload
"""
import os

os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

from app.core.config import settings
from app.core.vectorstore import VectorStoreManager


def upload_store_documents():
    """Upload store documents to mobile_docs collection"""
    print("📄 Uploading store documents...")

    try:
        # Document files and their topics
        doc_files = {
            "FAQ.docx": "FAQ часто задаваемые вопросы",
            "about.docx": "о магазине информация компании",
            "deals_sales.docx": "акции скидки распродажи предложения",
            "delivery.docx": "доставка курьер отправка",
            "installment.docx": "рассрочка кредит платежи installment",
            "stores_info.docx": "магазины адреса контакты филиалы",
            "warranty.docx": "гарантия warranty сервис ремонт"
        }

        documents = []
        docs_dir = settings.data_dir / "docs_dataset"

        # Process each document
        for filename, topic in doc_files.items():
            file_path = docs_dir / filename

            if file_path.exists():
                print(f"📖 Processing: {filename}")

                try:
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                    loaded_docs = loader.load()

                    for doc in loaded_docs:
                        content = doc.page_content.strip()
                        if content and len(content) > 50:
                            processed_doc = Document(
                                page_content=content,
                                metadata={
                                    "source": filename,
                                    "doc_type": Path(filename).stem,
                                    "topic": topic,
                                    "content_length": len(content)
                                }
                            )
                            documents.append(processed_doc)
                            print(f"   ✅ Added: {len(content)} chars")

                except Exception as e:
                    print(f"   ❌ Error processing {filename}: {e}")
            else:
                print(f"   ⚠️ File not found: {filename}")

        if not documents:
            print("❌ No documents processed!")
            return False

        # Upload to vector store
        print(f"\n📤 Uploading {len(documents)} documents...")

        vector_manager = VectorStoreManager()

        # Clear existing collection
        try:
            vector_manager.client.delete_collection(settings.docs_collection)
        except:
            pass

        # Create new collection
        from langchain_qdrant import QdrantVectorStore
        QdrantVectorStore.from_documents(
            documents,
            vector_manager.embeddings,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.docs_collection
        )

        print("✅ Store documents uploaded successfully!")

        # Show summary
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        print(f"\n📊 Upload Summary:")
        print(f"   Total documents: {len(documents)}")
        for doc_type, count in doc_types.items():
            print(f"   {doc_type}: {count} sections")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function"""
    print("🚀 O!Store Vector Store Setup")
    print("=" * 40)

    print("\n1️⃣ Checking configuration...")
    try:
        settings.validate()
        print("✅ Configuration valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    print("\n2️⃣ Uploading store documents...")
    if upload_store_documents():
        print("\n✅ Setup completed successfully!")
        print("🎯 You can now run the agent with: python main.py")
    else:
        print("\n❌ Setup failed!")


if __name__ == "__main__":
    main()