# Notes

- 30-04-2020:

    - Slide 2 (ma anche in seguito): quando ti riferisci a FCS e rCSI le chiami “endogenous data sources”, che e’ sicuramente corretto, pero’ secondo me per rendere piu’ chiaro quello che stiamo facendo io in questo contesto le definirei come “outcome variables” ovvero quello che vogliamo predire. Quando parli delle exogenous data sources nell’ambito del modello userei invece “covariates” o “features”. Nello specifico l’rCSI a seconda di come viene usato (come variabile da predire o come predictor per l’FCS) diventa, rispettivamente, un outcome variable o una covariate/feature (indipendentemente dal fatto che e’ esogena).
    - Slide 4: le threshold 21 e 35 in realta’ possono variare per alcuni paesi (e.g. nel middle east in realta’ se ne usano di diverse), quindi se le vuoi mettere, le metterei solo come esempio
    Farei una slide a parte sulla definizione precisa delle nostre output variables. Nel senso che spieghi molto bene come sono definiti FCS e rCSI, pero’ poi il fatto che la metrica precisa che usiamo noi e’ il % persone con boor + borderline diets e % persone con rCSI >=19 e’ solo accennato, tra altre cose, nella slide 6; invece e’ una cosa importante da chiarire e definire, quindi farei una slide a parte dopo la 4.
    - Slide 10: Analysis Data à Data Analysis (e in generale forse cambierei anche il titolo di quella sezione, ma al momento non mi viene un’idea precisa da suggerire 😉)
    - Slide 27: come ti accennavo anche durante la presentazione, non mi e’ chiarissimo il caso “shuffle”
    - Slide 29: come discusso, questo caso non e’ un multivariate forecasting, perche’ stai usando solo l’FCS; bisogna trovare un altro nome per definire questo caso, se no si fa confusione con quando fai effettivamente multivariate perche’ usi delle covariates diverse dall’FCS stesso.
    - Slide 34: come discusso, userei diversi colori per univariate e multivariate e per modelli statistici vs machine learning.
    
- 02-07-2020:

    - Per la nuova serie temporale del Ramadan considerare accumulo dato da una sliding window.
    - Box-plot: asse delle x ordinato dal valore più alto al più basso rispetto la mediana.
    - Fare la feature importance anche a lag maggiori (180 ad esempio).