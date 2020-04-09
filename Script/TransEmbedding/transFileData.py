"""rawDatasets  trans to (entry2id.txt,relation2id.txt,train.txt) """
import json
import os


def transFile(inputdir, outputdir):
    if not os.path.exists(outputdir): os.mkdir(outputdir)
    # entity2id.txt
    with open(inputdir + "/vocab/entity_vocab.json") as entityFile:
        entitydic = json.load(entityFile, encoding="utf-8")
    with open(outputdir + "/entity2id.txt", "w") as ef:
        ef.write(str(len(entitydic)) + "\n")
        for entity, id in entitydic.items():
            ef.write("{}\t{}\n".format(entity, id))
    # relation2id.txt
    with open(inputdir + "/vocab/relation_vocab.json") as relationFile:
        relationdic = json.load(relationFile, encoding="utf-8")
    with open(outputdir + "/relation2id.txt", "w") as rf:
        rf.write(str(len(relationdic)) + "\n")
        for relation, id in relationdic.items():
            rf.write("{}\t{}\n".format(relation, id))
    # train2id.txt
    trainfile = ["/graph.txt", "/train.txt", "/dev.txt", "/test.txt"]
    with open(outputdir + "/train2id.txt", "w") as tf:
        for file in trainfile:
            graphFile = open(inputdir + file, "r")
            lines = graphFile.readlines()
            tf.write(str(len(lines)) + "\n")
            for line in lines:
                line = line.replace("\n", "")
                line = line.split("\t")
                tf.write("{}\t{}\t{}\n".format(entitydic[line[0]], entitydic[line[2]], relationdic[line[1]]))
        graphFile.close()


if __name__ == '__main__':
    Names = ["AthleteHomeStadium/", "AthletePlaysForTeam/", "AthletePlaysInLeague/", "AthletePlaysSport/",
             "OrganizationHeadQuarteredInCity/", "OrganizationHiredPerson/", "PersonBornInLocation/",
             "PersonLeadsOrganization/", "TeamPlaysSport/", "WorksFor/","WN18RR"]
    datasetName = "AthletePlaysForTeam/"
    inputdir = "E:\AAAAA\ReforcementReasoning\RawDataSets/" + datasetName
    outputdir = "E:\AAAAA\ReforcementReasoning\PreDataSets/tranedFile/" + datasetName
    transFile(inputdir, outputdir)
