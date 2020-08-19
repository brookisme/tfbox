import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tfbox.metrics as metrics

EPS=1e-8

class ScoreKeeper(object):
    #
    # CONSTANTS
    #
    STRICT_ACCURACY='acc'


    #
    # PUBLIC
    #
    def __init__(self,
                 model,
                 loader,
                 classes,
                 nb_classes=None,
                 metric=STRICT_ACCURACY,
                 ignore_label=None,
                 row_keys=[],
                 band_names=[],
                 print_keys=[],
                 band_importance=True):
        self.model=model
        self.loader=loader
        self.classes=classes
        self.max_class_value=max(classes.keys())
        self.nb_classes=nb_classes or len(classes)
        self.row_keys=row_keys or []
        self.band_names=band_names or []
        self.nb_bands=len(self.band_names)
        self.band_importance=band_importance
        self._set_metric(metric,ignore_label)
        if print_keys is True:
            self.print_keys=[self.metric_name]
        elif print_keys:
            self.print_keys=print_keys+[self.metric_name]
        self.scores=None


    def report(self,frac=None,force=False):
        if force or (self.scores is None):
            nb_batches=len(self.loader)
            if not nb_batches:
                raise ValueError('no data to score')
            batch_indices=list(range(nb_batches))
            np.random.shuffle(batch_indices)
            if frac:
                batch_indices=batch_indices[:int(nb_batches*frac)]
            self.batch_indices=batch_indices
            self.report_length=len(batch_indices)
            self.scores=self._flatten([self.score_batch(b,i) for i,b  in enumerate(batch_indices)])
            self.scores=pd.DataFrame(self.scores)
            cols=self.row_keys+[c for c in self.scores.columns if c not in self.row_keys]
            self.scores=self.scores[cols]
        return self.scores
        

    def importance(self):
        if not self.band_importance:
            raise ValueError('band_importance must be set to True')
        if self.scores is None:
            raise ValueError('must run report before getting importance')
        else:
            icols=[f'importance_{self.band_names[i]}' for i in range(self.nb_bands)]
            return self.scores[icols].mean().sort_values()


    def score_batch(self,batch,index=None):
        inpts,targs=self.loader[batch]
        if self.row_keys:
            rows=self.loader.batch_rows
        else:
            rows=[False]*targs.shape[0]
        scores=[]
        importances=[]
        targs=tf.argmax(targs,axis=-1)
        preds=tf.argmax(self.model(inpts),axis=-1)
        for i,(targ,pred,row) in enumerate(zip(targs,preds,rows)):
            score_data={}
            score_data['batch']=batch
            score_data['image_index']=i
            if row is not False:
                for k in self.row_keys:
                    score_data[k]=row[k]
            score_data[self.metric_name]=self.metric(targ,pred).numpy()
            score_data=self.confusion(targ,pred,data=score_data)
            scores.append(score_data)
            self._print_score(score_data,index)
        if self.band_importance:
            if not self.nb_bands:
                self.nb_bands=inpts.shape[-1]
            for b in range(self.nb_bands):
                rpreds=tf.argmax(self.model(self._randomize(inpts,b)),axis=-1)
                if self.band_names:
                    b=self.band_names[b]
                for j,(targ,rpred) in enumerate(zip(targs,rpreds)):
                    base_score=scores[j][self.metric_name]
                    scores[j][f'importance_{b}']=self._importance(
                        targ,
                        rpred,
                        base_score)
        return scores


    def confusion(self,targ,pred,data={}):
        cm=tf.math.confusion_matrix(
            tf.reshape(targ,[-1]),
            tf.reshape(pred,[-1]),
            num_classes=self.nb_classes).numpy()
        row_cm=data.copy()
        for tk,tv in self.classes.items():
            for pk,pv in self.classes.items():
                row_cm[f'{tv}-{pv}']=cm[tk,pk]
        return row_cm


    
    #
    # INTERNAL
    #
    def _set_metric(self,metric,ignore_label):
        if metric and metric!=ScoreKeeper.STRICT_ACCURACY:
            if isinstance(metric,str):
                self.metric_name=metric
                self.metric=metrics.get(metric)
            else:
                self.metric=metric
                self.metric_name=metric.name
            try:
                self.metric=self.metric()
            except:
                pass
        else:
            self.metric_name=ScoreKeeper.STRICT_ACCURACY
            self.metric=self._acc_metric(ignore_label)


    def _acc_metric(self,ignore_label):
        def _acc(y_true,y_pred):
            if (ignore_label is None) or (ignore_label is False):
                valid=1
                total=tf.cast(tf.math.reduce_prod(y_true.shape),tf.float32)
            else:
                valid=tf.cast(y_true!=ignore_label,tf.float32)
                total=tf.reduce_sum(valid)
            valid_true=valid*tf.cast((y_true==y_pred),tf.float32)
            return tf.reduce_sum(valid_true)/total
        return _acc


    def _randomize(self,im,b):
        im=im.copy()
        bnd=tf.random.uniform(im.shape[:-1],maxval=self.max_class_value+1,dtype=tf.int32)
        if im.ndim==3:
            im[:,:,b]=bnd
        else:
            im[:,:,:,b]=bnd
        return im


    def _importance(self,targ,randomized_pred,base_score):
        return ((base_score-self.metric(targ,randomized_pred).numpy())/base_score+EPS)


    def _flatten(self,ll):
        return [i for l in ll for i in l]          


    def _print_score(self,data,index=None):
        if self.print_keys:
            parts=[f'{key}: {data[key]}' for key in self.print_keys]
            line=', '.join(parts)
            if index is not None:
                line=f'[{index}/{self.report_length}] {line}'
            print(line)



#
# NOTEBOOK DISPLAY
#
class ScoreBoard(object):
    """ a notebook parser for ScoreKeeper reports
    """
    FBETAS=[1,2,0.5]
    GROUP_KEY='data_split'
    GROUPS=['train','valid','test']

    #
    # STATIC
    #
    def precision_recall(tp,fx):
        return tp/(tp+fx+EPS)


    def fbeta(p,r,beta=2):
        return ((1+(beta**2))*p*r)/(((beta**2)*p)+r+EPS)


    #
    # PUBLIC
    #    
    def __init__(self,
            report,
            categories=[],
            fbetas=FBETAS,
            groups=GROUPS,
            group_key=GROUP_KEY,
            accuracy_key='acc',
            nb_display=print,
            drop_na=True):
        if isinstance(report,str):
            report=pd.read_csv(report)
        if drop_na:
            report.dropna(inplace=True)
        self.report=report
        self.categories=categories
        self.fbetas=fbetas
        self.group_key=group_key
        self.groups=groups
        self.accuracy_key=accuracy_key
        self.nb_display=nb_display


    def accuracy(self,key=None,value=None):
        df=self.report
        if key:
            df=df[df[key]==value]
        return df[self.accuracy_key].mean()


    def accuracies(self,key,sort=True,display=True,return_data=False):
        values=self.report[key].unique()
        _df=pd.DataFrame([{ 
            key:v, 
            self.accuracy_key: self.accuracy(key=key,value=v)} 
            for v in values])
        if sort:
            _df=_df.sort_values(self.accuracy_key)
        return self._display_return(_df,display,return_data)


    def stats(self,cm,cat,categories=None,fbetas=None):
        categories=categories or self.categories
        fbetas=fbetas or self.fbetas     
        not_cats=[c for c in categories if c!=cat]
        tp=cm[f'{cat}-{cat}']
        fp=cm[[f'{c}-{cat}' for c in not_cats]].sum()
        tn=cm[[f'{c}-{c2}' for c in not_cats for c2 in not_cats]].sum()
        fn=cm[[f'{cat}-{c}' for c in not_cats]].sum()
        p=ScoreBoard.precision_recall(tp,fp)
        r=ScoreBoard.precision_recall(tp,fn)
        _stats={
            'lulc':cat,
            'precision': p,
            'recall': r,
            'tp':tp,
            'fp':fp,
            'fn':fn,
            'tn':tn }
        for b in fbetas:
            bkey=re.sub("\.","",str(b))
            _stats[f'f_{bkey}']=ScoreBoard.fbeta(p,r,beta=b)
        return _stats


    def stat_report(self,
            categories=None,
            key=None,
            value=None,
            groups=None,
            group_key=None,
            fbetas=None,
            display=True,
            return_data=False):
        categories=categories or self.categories
        fbetas=fbetas or self.fbetas
        if group_key is False:
            groups=[None]
        else:
            group_key=group_key or self.group_key
            groups=groups or self.groups
        dfs=[]
        for group_value in groups:
            df=self.report.copy()
            if group_key:
                df=df[df[group_key]==group_value]
            if key:
                df=df[df[key]==value]
            cm=df[self._confusion_cols(categories)].sum()
            _df=pd.DataFrame([self.stats(cm,c,categories,fbetas=fbetas) for c in categories])
            if group_key:
                _df.loc[:,group_key]=group_value
                _df=_df[[group_key]+[k for k in _df.columns if k!=group_key]]
            if key:
                _df.loc[:,key]=value
                _df=_df[[key]+[k for k in _df.columns if k!=key]]
            dfs.append(_df)
        df=pd.concat(dfs)
        return self._display_return(df,display,return_data)


    def stat_reports(self,
            key,
            groups=None,
            group_key=None,
            categories=None,
            fbetas=None,
            display=True,
            return_data=False):
        categories=categories or self.categories
        fbetas=fbetas or self.fbetas
        values=self.report[key].unique().tolist()
        dfs=[]
        for v in values:
            _df=self.stat_report(
                categories=categories,
                key=key,
                value=v,
                group_key=group_key,
                groups=groups,
                display=False,
                return_data=True)
            dfs.append(_df)
        df=pd.concat(dfs)
        return self._display_return(df,display,return_data)


    #
    # INTERNAL
    #
    def _confusion_cols(self,categories):
        return [f'{t}-{p}' for t in categories for p in categories]


    def _display_return(self,out,display,return_data):
        if display and self.nb_display:
            self.nb_display(out)
        if return_data or (not display):
            return out

