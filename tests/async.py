import asyncio
import asyncpg

async def main():
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="amref",
            password="amref_secret",
            database="amref_helpdesk",
        )
        print("✅ Connected successfully!")
        print("Database:", await conn.fetchval("SELECT current_database()"))
        print("User:", await conn.fetchval("SELECT current_user"))
        await conn.close()
    except Exception as e:
        print(type(e).__name__)
        print(e)

asyncio.run(main())