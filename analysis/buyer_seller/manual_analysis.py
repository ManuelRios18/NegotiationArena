import os
import numpy as np
os.environ["OPENAI_API_KEY"] = "key"
from explorer.utils import load_states_from_dir


#log_dir_over_valued_prompted = ".logs/over_valued_buyer_self_interested_gpt-4o-2024-08-06-cde-aia/"
#log_dir_over_valued_prompted = ".logs/over_valued_buyer_self_interested/"
log_dir_over_valued_prompted = ".logs/over_valued_buyer_self_interested_gpt-4-turbo-2024-04-09-cde-aia_2"

def get_bad_proposal_rate(game_states):
    game_states = [ g for g in game_states if len(g.game_state) >=2]
    # extract trade object
    trade_proposals = [[ _['player_public_info_dict']['newly proposed trade'] for _ in  g.game_state[1:-1]] for g in game_states]
    # extract ZUP remove none
    trade_proposals = [[ _.resources_from_second_agent.resource_dict['ZUP'] if _ != "NONE" else 0 for _ in tp ] for tp in trade_proposals]
    # get sequences long enough
    trade_proposals = [tp for tp in trade_proposals if len(tp) >= 2]
    # get first two proposals
    trade_proposals = [tp[:2] for tp in trade_proposals]
    # flatten
    trade_proposals = np.array(trade_proposals)
    bad_proposal_rate=  np.average(trade_proposals[:,1] > trade_proposals[:, 0])
    trade_proposals, bad_proposal_rate
    return bad_proposal_rate


game_states_ov_prompted = load_states_from_dir(log_dir_over_valued_prompted)

over_valued_prompted_bpr = get_bad_proposal_rate(game_states_ov_prompted)
print(over_valued_prompted_bpr)
