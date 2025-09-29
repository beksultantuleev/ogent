"""
O!Store ReAct Agent - Main Entry Point
"""

from app.agent import OStoreAgent


def main():
    """Main CLI interface"""
    print("🚀 O!Store ReAct Agent")
    print("=" * 40)

    try:
        # Initialize agent
        print("🔧 Initializing agent...")
        agent = OStoreAgent()

        # Health check
        health = agent.health_check()
        if not health["vector_stores"]:
            print("❌ Vector stores not accessible")
            print("💡 Make sure Qdrant is running and collections exist")
            return

        print("✅ Agent initialized successfully")
        print("📝 Type your questions in Russian/Kyrgyz")
        print("🛑 Type 'exit' or 'quit' to stop")
        print("-" * 40)

        query_count = 0

        while True:
            user_input = input(f"\n👤 Вы: ")

            if user_input.lower() in ["exit", "quit", "выход"]:
                print("\n👋 До свидания! Спасибо за обращение в O!Store!")
                break

            query_count += 1
            print(f"\n🤖 Консультант ({query_count}):")

            try:
                response = agent.chat(user_input)
                print(response)
            except Exception as e:
                print(f"❌ Ошибка: {e}")

        # Show session stats
        stats = agent.logger.get_session_stats()
        if stats["total_sessions"] > 0:
            print(f"\n📊 Всего сессий: {stats['total_sessions']}")
            print(f"📊 Всего запросов: {stats['total_queries']}")

    except KeyboardInterrupt:
        print("\n\n👋 Остановлено пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка запуска: {e}")
        print("\n💡 Проверьте:")
        print("   1. Qdrant запущен (docker run -p 6333:6333 qdrant/qdrant)")
        print("   2. .env файл содержит OPENAI_API_KEY")
        print("   3. Коллекции существуют в Qdrant")


if __name__ == "__main__":
    main()