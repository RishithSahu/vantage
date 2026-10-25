"""DEV tool: trains the question classifier (vgpt/qa_clf.py) and writes vgpt/data/qa_clf.npz.
Training data: the fuzzy router's examples (vgpt/data/qa_protos.json, 'legacy' = questions the older pipeline answers), the labelled everyday sets, conversation-layer examples (smalltalk / Hindi-English / help) and
generated off-topic questions (class 'gate').
  python tools_train_clf.py            -> model WITHOUT heldout_dev / heldout_final (used to measure and to tune thresholds)
  python tools_train_clf.py --ship     -> model WITH heldout_dev and heldout_dev2 (heldout_final stays out, so its score remains a true test)"""
import json, os, random, sys
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vgpt.qa_clf import word_features, char_features

TOPICS = ["the capital of france", "a good pasta recipe", "the best movie of this year", "the bitcoin price", "the cricket score", "my maths homework", "a funny joke", "a poem about the sea", "the weather in paris", "how to lose weight", "the president of the usa", "a nice restaurant", "a song by queen",
          "the meaning of life", "how tall everest is", "dinosaurs", "the football world cup", "calories in a banana", "how to bake a cake", "my girlfriend", "a flight to delhi", "the stock market", "how to say hello in spanish", "the time in tokyo", "a good book to read", "who painted the mona lisa",
          "the news today", "a workout plan", "my horoscope", "the lyrics of a song", "how to tie a tie", "the population of india", "a chess opening", "how to learn python", "tips for a job interview", "a birthday gift idea", "the tallest building", "a bedtime story", "how photosynthesis works", "the speed of light"]
TOPICS += ["how to make biryani", "a recipe for noodles", "how to cook sushi at home", "the best burger place", "the weather in mumbai", "the weather in new york", "the weather in dubai tomorrow", "who is the prime minister of india", "who is the chancellor of germany", "something funny", "a comedian", "a stand up show", "the latest bollywood movie", "the ipl match", "a cricket player", "the exchange rate", "a mobile phone to buy", "the best laptop", "what 12 times 13 is", "the answer to 45 plus 67", "my phone not charging", "my laptop battery draining fast", "my tv remote not working", "the square root of 144", "how many days until christmas", "the thing doing the other thing", "something about nothing in particular", "when the sun sets", "the distance to the moon", "how many planets there are", "the history of rome", "the best way to learn guitar", "a good name for a dog", "the score of the match", "a quote about life", "how to cook rice", "the tallest mountain", "the largest ocean", "how to meditate",
           "a travel plan for goa", "the price of gold", "my exam results", "the latest smartphone", "how to write a cv", "the rules of tennis", "a good gym", "the capital of japan", "how many days in a leap year", "who invented the telephone", "how to grow tomatoes", "how to make coffee",
           "who is the richest man", "a funny story", "what time the shops close", "directions to the nearest hospital", "my bank balance", "how to make tea", "the lyrics of that song", "the biggest animal", "a science fact", "how rainbows form", "the oldest city", "a good podcast"]
TPL = ["tell me about {}", "what is {}", "can you give me {}", "i want to know about {}", "{}", "do you know {}", "please find {}", "show me {}", "help me with {}", "i need {}"]
SMALL = ["hi", "hello", "hey", "hey there", "hi there", "hello assistant", "good morning", "thanks", "thank you", "thank you so much", "thanks a lot", "cheers", "ok thanks", "ok thank you", "bye", "goodbye", "see you", "good night", "good job", "well done", "great", "nice", "awesome", "perfect", "cool", "ok", "okay", "alright", "sorry", "how are you", "whats up"]
FOREIGN = ["gaadi kaisi hai", "gaadi ka haal batao", "kitni battery bachi hai", "range kitna bacha hai", "bhai kya bol rahe ho bhai", "gaadi ka kya haal hai", "gaadi theek hai kya", "battery kitni bachi hai", "range kitna hai", "kitna chalegi gaadi", "mujhe batao battery kitni hai", "service kab hai", "gaadi safe hai kya", "kya haal hai", "kuch problem hai kya", "yaar gaadi kaisi hai", "bhai range batao", "tyre mein hawa kitni hai", "kitna charge hai"]
HELP = ["how do i use this assistant", "how does this assistant work", "what is this app for", "what can you do", "what can i ask you", "help", "help me", "how do i use you", "show me examples", "what questions can i ask", "what are you able to do", "what topics do you cover", "what do you know", "list your features", "give me sample questions", "i dont know what to ask"]


def load(fn):
    out = []
    p = os.path.join(HERE, fn)
    if not os.path.exists(p): return out
    for l in open(p, encoding="utf-8"):
        if l.strip() and not l.startswith("#") and "\t" in l: out.append(tuple(l.rstrip("\n").split("\t", 1)))
    return out


def main():
    ship = "--ship" in sys.argv; X, y = [], []
    for r, qs in json.load(open(os.path.join(HERE, "vgpt", "data", "qa_protos.json"), encoding="utf-8")).items():
        for q in qs: X.append(q); y.append("legacy" if r == "_legacy" else r)
    files = ["bench/everyday_blind.tsv", "bench/everyday_blind2.tsv", "bench/everyday_blind3.tsv", "bench/everyday_blind4.tsv"] + (["bench/heldout_dev.tsv", "bench/heldout_dev2.tsv"] if ship else [])
    for fn in files:
        for lab, q in load(fn): X.append(q); y.append(lab)
    wts = [1.0] * len(X)
    for lab, q in load("bench/everyday_grammar.tsv"): X.append(q); y.append(lab); wts.append(0.35)                      # generated wordings count less than hand-written ones
    for t in TOPICS:
        for p in TPL: X.append(p.format(t)); y.append("gate")
    for q in SMALL: X.append(q); y.append("smalltalk")
    for q in FOREIGN: X.append(q); y.append("foreign")
    for q in HELP: X.append(q); y.append("help")
    wts += [1.0] * (len(X) - len(wts))
    # soh_rate / battery_age are one idea

    vw = TfidfVectorizer(analyzer=word_features, sublinear_tf=True, lowercase=False); vc = TfidfVectorizer(analyzer=char_features, sublinear_tf=True, lowercase=False, min_df=2)
    A = hstack([vw.fit_transform(X), vc.fit_transform(X)]).tocsr()
    clf = LinearSVC(C=1.0).fit(A, y, sample_weight=np.array(wts))
    # prune tiny weights so the file stays small
    W = clf.coef_.astype(np.float32); W[np.abs(W) < 1e-3] = 0.0
    wv = [None] * len(vw.vocabulary_); [wv.__setitem__(i, w) for w, i in vw.vocabulary_.items()]
    cv = [None] * len(vc.vocabulary_); [cv.__setitem__(i, w) for w, i in vc.vocabulary_.items()]
    out = os.path.join(HERE, "vgpt", "data", "qa_clf.npz")
    np.savez_compressed(out, classes=np.array(clf.classes_), W=W.astype(np.float16), b=clf.intercept_.astype(np.float32), w_vocab=np.array(wv), c_vocab=np.array(cv), w_idf=vw.idf_.astype(np.float32), c_idf=vc.idf_.astype(np.float32))
    print(f"{len(X)} examples, {len(clf.classes_)} classes, {len(wv)} word + {len(cv)} char features -> {out} ({os.path.getsize(out) / 1e6:.1f} MB){' [shipping model: includes heldout_dev]' if ship else ''}")


if __name__ == "__main__":
    main()
