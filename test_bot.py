import uuid
from langchain_core.messages import HumanMessage
# IMPORT the graph and greeting directly from your main application!
from main import app_graph, SYSTEM_PROMPT


def run_cli():
    print("==================================================")
    print("  DineLine AI - Local Terminal Testing Simulator  ")
    print("  Type 'quit' or 'exit' to stop the simulator.    ")
    print("==================================================\n")

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    greeting = "Welcome to DineLine Pizza! What would you like to order today?"
    print(f"AI: {greeting}")

    is_first_turn = True

    while True:
        try:
            user_input = input("\nYou: ")

            if user_input.lower() in ['quit', 'exit']:
                print("Exiting simulator...")
                break
            if not user_input.strip():
                continue

            if is_first_turn:
                input_data = {
                    "messages": [
                        {"role": "assistant", "content": greeting},
                        {"role": "user", "content": user_input}
                    ],
                    "cart": [],
                    "order_total": 0.0,
                    "requires_handoff": False
                }
                is_first_turn = False
            else:
                input_data = {"messages": [{"role": "user", "content": user_input}]}

            # Process through the LangGraph imported from main.py
            events = app_graph.invoke(input_data, config=config)

            # --- FIX: Parse the weird Gemini list output into clean text ---
            raw_content = events["messages"][-1].content
            if isinstance(raw_content, list):
                # If Gemini returns a list of dictionaries, extract just the text
                ai_response = " ".join([block["text"] for block in raw_content if block.get("type") == "text"])
            else:
                ai_response = raw_content

            cart = events.get("cart", [])
            total = events.get("order_total", 0.0)
            requires_handoff = events.get("requires_handoff", False)

            print(f"AI: {ai_response}")
            print(f"\n[DEBUG STATE] -> Cart: {cart} | Total: ${total} | Handoff Triggered: {requires_handoff}")

            if requires_handoff:
                print("\n*** HUMAN HANDOFF INITIATED. SIMULATOR ENDING ***")
                break

        except KeyboardInterrupt:
            print("\nExiting simulator...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")


if __name__ == "__main__":
    run_cli()