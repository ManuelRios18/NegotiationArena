import os
from dotenv import load_dotenv
from negotiationarena.game_objects.resource import Resources
from negotiationarena.agents.azure_chatgpt import AzureChatGPTAgent
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
import traceback
from games.buy_sell_game.game import BuySellGame


load_dotenv(".env")


if __name__ == "__main__":
    for i in range(1):
        try:
            a1 = AzureChatGPTAgent(
                agent_name=AGENT_ONE,
                model="gpt-4o-2024-08-06-cde-aia",
                azure_endpoint=os.getenv("OPENAI_API_BASE_2"),
                api_key=os.getenv("OPENAI_API_KEY_2"),
                api_version=os.getenv("OPENAI_API_VERSION_2")
            )
            a2 = AzureChatGPTAgent(
                agent_name=AGENT_TWO,
                model="gpt-4o-2024-08-06-cde-aia",
                azure_endpoint=os.getenv("OPENAI_API_BASE_2"),
                api_key=os.getenv("OPENAI_API_KEY_2"),
                api_version=os.getenv("OPENAI_API_VERSION_2")

            )

            c = BuySellGame(
                players=[a1, a2],
                iterations=10,
                player_goals=[
                    SellerGoal(cost_of_production=Valuation({"X": 40})),
                    BuyerGoal(willingness_to_pay=Valuation({"X": 60})),
                ],
                player_starting_resources=[
                    Resources({"X": 1}),
                    Resources({MONEY_TOKEN: 1000}),
                ],
                player_conversation_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                player_social_behaviour=[
                    "",
                    "You are very kind and generous. Be friendly and helpful "
                    "with the other player, they are your dearest friend.",
                ],
                log_dir="../example_logs_ignore/buysell",
            )

            c.run()
        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            stack_trace = traceback.format_exc()

            # Print or use the information as needed
            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Stack Trace:\n{stack_trace}")
