import xml.etree.ElementTree as ET
import collections
import os
class Biosent():
    def __init__(self):
        self.words = None

def get_offs(charOffset):
    ranges = charOffset.split(';')
    par = '-' if '-' in charOffset else ' '
    print(ranges)
    offsets = (ranges[0].split(par)[0]+'-'+ranges[0].split(par)[1]) if len(ranges)==1 else ranges[0].split(par)[0]+'-'+ranges[-1].split(par)[1]
    return offsets.split('-')

def parse_annfile(path,fchardict,bchardict):
    #returns spanid,(start,end)
    fl = open(path,'r')
    tid2offsets = {}
    spanid  = 0
    clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for line in fl:
        annots = line.strip().split()
        if annots[0][0]=='T':
            print(path)
            annots = line.strip().split('\t')
            tid = annots[0]
            print(annots)
            fcharoff,bcharoff = get_offs(' '.join(annots[1].split(' ')[1:]))
            try:
                tid2offsets[tid] = (fchardict[fcharoff],bchardict[bcharoff])
            except Exception:
                continue
        else:
            try :
                for i in range(2,4):
                    clusters[spanid].append(tid2offsets[annots[i].split(':')[1]])
                spanid+=1
            except Exception:
                continue
    return clusters



def get_partition(root):
    ntok=0
    fchardict = {}
    bchardict = {}
    partition = {}
    for child1 in root:
        if child1.tag!='sentence':
            continue
        for child2 in child1:
            if child2.tag!='tokens':
                continue
            for child3 in child2:
                fcharoff,bcharoff = tuple(child3.attrib['charOffset'].split('-'))
                fchardict[fcharoff] = (fcharoff,bcharoff,child3.attrib['text'])
                bchardict[bcharoff] = (fcharoff,bcharoff,child3.attrib['text'])

    for child1 in root:
        if child1.tag!='Term':
            continue
        try:
            fcharoff,bcharoff = get_offs(child1.attrib['charOffset'])
            if fchardict.get(fcharoff) is None or bchardict.get(bcharoff) is None:
                print('None')
                if fchardict.get(fcharoff) is None:
                    foff,boff,txt = bchardict[bcharoff]
                    print(foff,boff,txt)
                    print(fcharoff)
                    print((txt[0:int(fcharoff)-int(foff)-1],txt[int(fcharoff)-int(foff):int(boff)-int(foff)]))
                    partition[(foff,boff)] = (txt[0:int(fcharoff)-int(foff)-1],txt[int(fcharoff)-int(foff):int(boff)-int(foff)])
                if bchardict.get(bcharoff) is None:
                    foff,boff,txt = fchardict[fcharoff]
                    partition[(foff,boff)] = (txt[0:int(bcharoff)-int(foff)],txt[int(bcharoff)-int(foff)+1:int(boff)-int(foff)])
        except Exception:
            continue
    print(partition)
    return partition

class Biomed():
    def __init__(self,xml_dir_path,ann_dir_path):
        self.xmllst = [os.path.join(xml_dir_path,pth) for pth in sorted(os.listdir(xml_dir_path))]
        self.annlst = [os.path.join(ann_dir_path,pth) for pth in sorted(os.listdir(ann_dir_path)) if pth.endswith("ann")]
        # print(self.xmllst,self.annlst)
        
    def dataset_document_iterator(self):
        #returns the document with 
        for xml_path,ann_path in  zip(self.xmllst,self.annlst):
            tree = ET.parse(xml_path)
            print(xml_path)
            root = tree.getroot()
            fchardict = {}
            bchardict = {}
            #looping over all the sentences.
            sents = []
            partition = get_partition(root)
            for idx1,child1 in enumerate(root):
                if child1.tag != "sentence":
                    continue

                #looping over all the tokens.
                sent = Biosent()
                words = []
                ntok = 0
                for child2 in child1:
                    if child2.tag == "tokens":
                        for idx3,child3 in enumerate(child2):
                            fcharoff,bcharoff = tuple(child3.attrib["charOffset"].split('-'))
                            if partition.get((fcharoff,bcharoff)) is None:
                                words.append(child3.attrib['text'])
                                fchardict[fcharoff] = ntok
                                bchardict[bcharoff] = ntok+1
                                ntok+=1
                            else:
                                wordl = partition.get((fcharoff,bcharoff))
                                initoff = int(fcharoff)
                                for wid,word in enumerate(wordl):
                                    words.append(word)
                                    off = len(word)
                                    fchardict[str(initoff)] = ntok
                                    bchardict[str(int(initoff)+off)] = ntok+1
                                    ntok+=1
                                    initoff+=off+(1 if wid==0 else 0)
                sent.words = words
                sents.append(sent)
            clusters = parse_annfile(ann_path,fchardict,bchardict)
            yield sents,clusters


                    