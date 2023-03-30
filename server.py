from concurrent import futures
import logging

import grpc
import sound_pb2
import sound_pb2_grpc
import os
import pdb
import sys

#add working directory to run model
sys.path.insert(1, '/home/solution-python2/model/EfficientAT')
os.chdir("./model/EfficientAT")
from model.EfficientAT.inference import inference
os.chdir("../..")

#using defined proto, run model
class File(sound_pb2_grpc.FileServicer):
	
    #define function, analizing the sound file.
    def Define(self, request, context):
        #save sound file sended from main server
        f = open('./model/EfficientAT/resources/temp.wav','wb')
        f.write(request.sound)
        f.close()

        #analize the sound file
        temp=inference('./model/EfficientAT/resources/temp.wav')
        print(temp)

        #analize the result, show the most likely tag
        keys = list(temp.keys())
        maxpercent=0.0
        result = keys[0]
        for key in keys:
            if temp[key]>maxpercent:
                result = key
                maxpercent = temp[key]
        
        #using tagging rate, define if we have to alarm client
        percent = temp[result]
        if result=="Car horn":
            if percent>0.1:
                alarm=True
            else:
                alarm=False
        else:
            if percent>0.15:
                alarm=True
            else:
                alarm=False
        
        #send the result to main server
        return sound_pb2.SoundResponse(alarm = alarm,res=result, tagging_rate=percent)

    #check ping pong from main server
    def Connect(self, request, context):
        print(request.ping)
        return sound_pb2.Pong(pong='%s Pong!' % request.ping)

#open server with port 8080
def serve():
    #while analizing the soundfile, other soundfile could overlap the saved soundfile.
    #so only one thread can execute
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    sound_pb2_grpc.add_FileServicer_to_server(File(), server)
    server.add_insecure_port('[::]:8080')
    print("server on port 8080")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()