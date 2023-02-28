from data.input_behavior.prep_activity_profile.source.markov_model import MarkovActivityProfileGenerator
from data.input_behavior.prep_activity_profile.source.tou_processor import TouActivityProfileProcessor
from data.input_behavior.prep_activity_profile.source.commuting_time import CommutingTimer


def process_tou_activity_profile():
    tapp = TouActivityProfileProcessor()
    tapp.convert_tou_activity_profile()
    tapp.generate_input_activity_profile()
    tapp.plot_activity_share()


def gen_markov_matrix():
    model = MarkovActivityProfileGenerator()
    model.generate_activity_change_prob_matrix()
    model.generate_activity_duration_prob_matrix()


def gen_commuting_time():
    ct = CommutingTimer()
    ct.commuting_time()


if __name__ == "__main__":
    # process_tou_activity_profile()
    # gen_markov_matrix()
    gen_commuting_time()
