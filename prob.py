from __future__ import division
import json
import pickle

def probability(so, tags, post_frame):
    s_prob_t = {}
    for tag in tags:
        s_prob = {}
        count = 0
        for word in so:
            word = word[0]
            s_prob[word] = 0
            for i in range(len(post_frame)):
                if word in post_frame[0][i] and tag in post_frame[1][i]:
                    s_prob[word] = 1
                    count += 1
                    break
        if count > 0:
            s_prob_t[tag] = count/len(s_prob)
    return s_prob_t

def getTotalProbability(tot_so, tags, soNtags):
    probs = []
    for i, so in enumerate(tot_so):
        print "\rProcessing %d" % i
        p = probability(so, tags, soNtags)
        probs.append(p)
    return probs

def main():
    tot_so = json.load(open('postags.json', 'r'))
    tags = pickle.load(open('tags.pkl', 'rb'))
    soNtags = pickle.load(open('soNtags.pkl', 'rb'))
    print "Loading completed"
    print "Total: %d" % len(tot_so) 
    print ""
    tot_prob = getTotalProbability(tot_so, tags, soNtags)
    pickle.dump(tot_prob, open("probabilities.pkl", "wb"))
    print "Completed"


if __name__ == '__main__':
    main()