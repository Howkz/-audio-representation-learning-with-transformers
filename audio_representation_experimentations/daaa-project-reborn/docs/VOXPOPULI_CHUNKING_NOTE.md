# VoxPopuli et chunking - Note technique

## Pourquoi cette note existe

Les consignes du TP mentionnent explicitement VoxPopuli et la question du chunking
pour les signaux plus longs. Nous n'avons pas fait de campagne VoxPopuli complete
dans la voie finale, mais il faut documenter clairement pourquoi, et ce qu'une
implementation propre devrait faire.

## Probleme de fond

Pour l'ASR, il ne suffit pas de :

- couper brutalement un audio long ;
- garder la transcription complete.

Cette pratique cree une supervision incoherente :

- l'audio montre seulement un segment ;
- la cible textuelle decrit l'enonce complet.

Le TP pousse implicitement vers une autre logique :

- filtrer les durees quand on veut rester simple ;
- ou bien chunker proprement les longues sequences.

## Ce qu'un chunking propre devrait garantir

1. Coherence audio / texte

- un chunk audio ne doit pas etre associe a une transcription qui depasse largement
  ce qui est present dans le segment ;
- si l'alignement mot-audio n'est pas disponible, il faut rester prudent et eviter
  des chunkings agressifs.

2. Longueurs bornees

- chunks de duree bornes pour la VRAM ;
- padding et batching plus simples ;
- longueurs CTC plus stables.

3. Recouvrement controle

- si on chunk un long signal sans alignement fin, un recouvrement partiel peut
  limiter la perte d'information a la frontiere ;
- mais il faut alors eviter de compter plusieurs fois les memes references dans
  l'evaluation.

4. Evaluation honnete

- les predictions chunk-level doivent etre recomposees proprement ;
- ou bien l'evaluation doit rester explicitement chunk-level.

## Pourquoi ce n'est pas dans la voie finale retenue

La voie finale Phase 7 a privilegie LibriSpeech `clean` avec filtrage de duree
plutot qu'un vrai protocole VoxPopuli, pour trois raisons :

1. stabiliser d'abord la chaine ASR ;
2. rester dans les contraintes memoire/disque ;
3. eviter d'ajouter une complexite data supplementaire avant d'avoir une base
   distillee defensable.

## Ce qui a ete fait a la place

Dans la voie Phase 7 :

- les splits ASR critiques utilisent `length_policy: none` ;
- un garde-fou refuse les configurations incoherentes qui cropent un audio tout
  en gardant la transcription complete ;
- la simplification se fait par filtrage de duree et de longueur de transcript,
  pas par chunking naif.

## Ce qu'il faudrait implementer si on rouvre VoxPopuli

Minimum credible :

1. definir une duree cible de chunk ;
2. segmenter les audios longs avec recouvrement leger ;
3. disposer d'une strategie explicite pour l'association texte/chunk ;
4. documenter le protocole d'evaluation ;
5. comparer ce protocole a une baseline filtree plus simple.

## Conclusion

Le chunking VoxPopuli n'est pas abandonne parce qu'il serait inutile.
Il est repousse parce qu'une implementation rapide et naive casserait plus
facilement la supervision qu'elle n'ameliorerait l'ASR.

Pour le rapport, la position defendable est :

- la question a ete identifiee correctement ;
- la voie finale a choisi une simplification plus sure ;
- un protocole chunking propre reste un prolongement naturel si du budget
  experimental supplementaire existe.
