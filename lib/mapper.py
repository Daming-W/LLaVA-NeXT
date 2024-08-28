import os 
from tqdm import tqdm

class Mapper_old():
    def __init__(self,answer_dict):
        self.answer_dict = answer_dict

        self.bool_dict = {
            "smoking": False,
            "hailing": False,
            "ped_on_lawn": False,
            "crowded": False,
            "fire": False,
            "trash": False,
            "illegal_parking": False,
        }

        # seven predefined scenarios
        self.set_smoking = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_hailing = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_ped_on_lawn = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_crowded = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_fire = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_trash = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_illegal_parking = set([
            "True", "true", "Yes", "yes",
            ])
    
    def sentence2words(self,answer):
        return [word.strip(".,!?") 
                for word in answer.split() 
                if word.strip(".,!?;").isalpha()]

    def answer2bool(self):

        self.bool_dict["smoking"] = any(word in self.set_smoking for word in self.sentence2words(self.answer_dict["smoking"]))

        self.bool_dict["hailing"] = any(word in self.set_hailing for word in self.sentence2words(self.answer_dict["hailing"]))

        self.bool_dict["ped_on_lawn"] = any(word in self.set_ped_on_lawn for word in self.sentence2words(self.answer_dict["ped_on_lawn"]))

        self.bool_dict["crowd"] = any(word in self.set_crowded for word in self.sentence2words(self.answer_dict["crowded"]))

        self.bool_dict["fire"] = any(word in self.set_fire for word in self.sentence2words(self.answer_dict["fire"]))
        
        self.bool_dict["trash"] = any(word in self.set_trash for word in self.sentence2words(self.answer_dict["trash"]))

        self.bool_dict["illegal_parking"] =  any(word in self.set_illegal_parking for word in self.sentence2words(self.answer_dict["illegal_parking"]))

        return self.bool_dict
        

class Mapper():

    def __init__(self,answer_dict):
        # input answer sentence dict
        self.answer_dict = answer_dict

        # output boolean dict
        self.bool_dict = {key: None for key in self.answer_dict}

        # keyword for boolean convertions
        self.keyword_dict = set(['True','true','Yes','yes'])

        # define seven scenarios
        self.scenarios = {
            "smoking" : ["smoking"],
            "hailing" : ["hailing"],
            "ped_on_lawn" : ["ped_on_lawn"],
            "crowd" : ["crowd", "tent", "destruction", "crowded"],
            "fire_or_flood" : ["fire", "flood", "fire_or_flood"],
            "trash_or_fallen_leaves" : ["trash", "fallen_leaves", "trash_or_fallen_leaves"],
            "illegal_parking" : ["illegal_parking"],
            "faint" : ["faint"]
        }
        # output dict
        self.output_dict = {key: False for key in self.scenarios}
    
    def sentence2words(self,answer):
        return [word.strip(".,!?") 
                for word in answer.split() 
                if word.strip(".,!?;").isalpha()]

    def answer2bool(self):

        for sce, answer in self.answer_dict.items():

            self.bool_dict[sce] = any(word in self.keyword_dict for word in self.sentence2words(answer))

        return self.bool_dict

    def merge_bool(self):
        
        # check bool dict first
        if None in self.bool_dict.values():
            print('WARN: None in bool_dict! To use answer2bool to compute bool_dict first!')
            return None
        # convert by predefined scenarios
        for sce, sub_sce_list in self.scenarios.items():
            for sub_sce in sub_sce_list:
                if sub_sce in self.bool_dict:
                    if self.bool_dict[sub_sce] is True:
                        self.output_dict[sce] = True
    
        return self.output_dict
        
        
if __name__=="__main__":
    mapper = mapper(answer="True, it is on fire")
