import json

class InformationSystem:
    def __init__(self):
        pass
    
    
    def __extract_CSDL(self, rs_filename):
        jf = open(rs_filename)
        data = json.load(jf)
        jf.close()
        return data
    
    def information_retrieval(self, name):
        data = self.__extract_CSDL("Data/csdl.json")
        for key, value in data.items():
            if key == name:
                return value
        return None
    
    def main(self, name):
        return self.information_retrieval(name)
    
