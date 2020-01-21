
# Good News Everyone:  A Corpus of News Headlines Annotated with Emotions, Semantic Roles and Reader Perception

This `.zip` file contains the GoodNewsEveryone! dataset which we will make public upon
acceptance of the paper under review in a Git repository. This data is currently preliminary.
It contains the adjudicated dataset only. We will release all annotations once the paper is being accepted at LREC 2020.


## Reference

If used (after acceptance), please cite:

```
  @inprocedings{Bostan2020,
      author = {Laura Bostan, Evgeny Kim, Roman Klinger},
      title = {Good News Everyone: A Corpus of News Headlines Annotated with \\ Emotions, Semantic Roles and Reader Perception},
      booktitle = {Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)},
      year = {2020},
      month = {may},
      date = {11-16},
      pdf = {http://www.lrec-conf.org/proceedings/lrec2020/pdf/},
      language = {english},
      location = {Marseille, France},
      url = {https://www.aclweb.org/anthology/}
  }
```

## Content
The file `release_review.jsonl` contains the adjudicated annotated dataset.

Each line is a JSON encoded document, representing a single headline and its
annotations. The top level contains the ``headline`` itself, an object containing
``meta`` information (such as the ``source``, the ``country``, and the ``bias`` according to
the Media Bias Chart), as well as the most important part: the ``annotations``
object. Per type of annotation, this contains an object again, e.g.
``most_dominant`` for the annotation of the most dominant emotion, ``other_emotions``,
``reader_emotions``, ``intensity``, ``cue``, ``experiencer``, ``cause`` and ``target``.

These objects contain only the key ``gold`` (but will contain ``raw`` data in the final
release), which is either a single atom, or a list of annotations deemed
correct.

### Example:

~~~json
{
"headline": "Dan Crenshaw slams Chuck Schumer for ‘immature and deeply cynical’ reaction to the deal with Mexico",
"meta": {
    "phase1_rank": 4,
    "source": "Twitchy",
    "country": "US",
    "bias": {
      "vertical": "14",
      "horizontal": "29"
    },
"annotations": {
    "dominant_emotion": {
      "gold": "anger"
    },
    "other_emotions": {
      "gold": "annoyance"
    },
    "reader_emotions": {
      "gold": "annoyance"
    },
    "intensity": {
      "gold": "medium"
    },
    "cause": {
      "gold": [
        [
          "‘immature and deeply cynical’ reaction to the deal with mexico"
        ]
      ]
    },
    "cue": {
      "gold": [
        [
          "slams"
        ]
      ]
    "experiencer": {
      "gold": [
        [
          "dan crenshaw"
        ]
      ]
    },
    "target": {
      "gold": [
        [
          "chuck schumer"
        ]
      ]
    },
   }
 }
~~~

----

## Contact
Laura Ana Maria Bostan: laura.bostan@ims.uni-stuttgart.de
Evgeny Kim: evgeny.kim@ims.uni-stuttgart.de
Roman Klinger: roman.klinger@ims.uni-stuttgart.de


