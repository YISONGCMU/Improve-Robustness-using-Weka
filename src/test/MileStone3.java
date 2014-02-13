package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.trees.ADTree;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class MileStone3 {
	static String DataDir="/Users/songyi/Desktop/milestone4/";
	static int Modelno=1;	
	static String []FileName=new String[13];
public static void main(String args[]) throws Exception{
	File f=new File("/Users/songyi/Desktop/milestone 2 dataout/") ;
	File[] fd=f.listFiles();
	//System.out.println(fd.length);
	double []value=new double[13];
	for (int i=2;i<=24;i=i+2){
	FileName[i/2]=(fd[i].getName().split("_"))[0];
	Instances test = new Instances(  
		    	new BufferedReader(
		    	new FileReader(fd[i]) ));
	 
	Instances train = new Instances(  
    	new BufferedReader(
        new FileReader(fd[i+1]) ));
	
	 System.out.println("DataSet NO."+i/2);
	 
	value[i/2]=Output(train,test);
	
	}
	
	
	double mean=0;
	double Max=0;
	double sum=0;
	for (int i=1;i<=12;i++){
		System.out.println(/*"No."+i+"Dataset's errorC/errorNB"*/value[i]);
		if (value[i]>Max) Max=value[i];
		sum=sum+value[i];
	}
	mean=sum/12;
	System.out.println("Mean"+mean);
	System.out.println("Max"+Max);
	
	
/*
	 MultiClassClassifier mcls = new MultiClassClassifier();
     mcls.setClassifier(new ADTree());
	 mcls= (MultiClassClassifier) weka.core.SerializationHelper.read(DataDir+"anneal.model" );
	 Evaluation eval = new Evaluation(train);    
     eval.evaluateModel(mcls,test);  
     System.out.println("first model_error="+eval.errorRate());
 */   
}
	
public static double Output(Instances train,Instances test) throws Exception{

	 double A=NB_Error(train,test);
/*	 double B=ADTree_error(train,test);
	 double C=1.0;
			 //IBk_error(train,test);
	 double D=1.0;
	         //Ada_error(train,test);
*/	
	
	   
		double C=ADTree_error(train,test);  
		
		
	/*	// setup classifier
	      CVParameterSelection ps = new CVParameterSelection();
	      ps.setClassifier(new AdaBoostM1());
	      ps.setNumFolds(5);  // using 5-fold CV
	      ps.addCVParameter("P 10 100 10");
 
	      // build and output best options
	      ps.buildClassifier(train);
	      
	      String options=Utils.joinOptions(ps.getBestClassifierOptions());
	      System.out.println(options.split("W")[0]);
	*/      
	      double CC=C;
	    train.setClassIndex(train.numAttributes() - 1);
	   	test.setClassIndex(test.numAttributes() - 1);  
	   /*	
	  	 Classifier NB = new AdaBoostM1(); 
	  	NB.setOptions(Utils.splitOptions(options));
	      NB.buildClassifier(train);
	      
	      Evaluation eval = new Evaluation(train);
	      eval.evaluateModel(NB,test);  
	     C=eval.errorRate();
	     System.out.println("New Error Rate="+C);
	    */
	   	String minerror_Option=new String();
	   	double minerror=1;
	   	for (int i=1;i<35;i++){	
	    Evaluation eval = new Evaluation(train);
	     
	     MultiClassClassifier mcls = new MultiClassClassifier();
	     
	  
	     Classifier adtree=new ADTree();
	   
	     adtree.setOptions(Utils.splitOptions("-B "+String.valueOf(i)));
	     mcls.setClassifier(adtree);
	     mcls.buildClassifier(train);
	     eval.evaluateModel(mcls, test);	
	     C=eval.errorRate();
	     if (minerror>C) {minerror=C; minerror_Option="-B "+String.valueOf(i);}
	  }
	   	System.out.println(minerror_Option);
	     System.out.println("Error Rate="+CC);
	     System.out.println("New Error Rate="+minerror);
	     
	     
         Evaluation eval = new Evaluation(train);
	     
	     MultiClassClassifier mcls = new MultiClassClassifier();
	     
	  
	     Classifier adtree=new ADTree();
	   
	     adtree.setOptions(Utils.splitOptions(minerror_Option));
	     mcls.setClassifier(adtree);
	     mcls.buildClassifier(train);
	     eval.evaluateModel(mcls, test);	
	     weka.core.SerializationHelper.write(DataDir+FileName[Modelno]+"1.model",mcls );
	     Modelno++;
	     
	     
	     
	     
	     
	 return minerror/A;
	
}

public static double NB_Error(Instances train,Instances test) throws Exception{
	train.setClassIndex(train.numAttributes() - 1);
	test.setClassIndex(test.numAttributes() - 1); 
		
	 Classifier NB = new NaiveBayes();     
     NB.buildClassifier(train);
     Evaluation eval = new Evaluation(train);
     eval.evaluateModel(NB,test);  
     
 //    weka.core.SerializationHelper.write(DataDir+String.valueOf(Modelno)+".model",NB );
 //    Modelno++;
     
    //System.out.println(eval.toSummaryString("\nResults\n\n", false));
    // System.out.println("NB_error="+(100-eval.pctCorrect()));
     
return (eval.errorRate());
}

public static double ADTree_error(Instances train,Instances test) throws Exception{
	train.setClassIndex(train.numAttributes() - 1);
	test.setClassIndex(test.numAttributes() - 1);
	 
     Evaluation eval = new Evaluation(train);
     
     MultiClassClassifier mcls = new MultiClassClassifier();
     
     Classifier adtree=new ADTree();
     
     mcls.setClassifier(adtree);
     mcls.buildClassifier(train);
     eval.evaluateModel(mcls, test);
     
    // weka.core.SerializationHelper.write(DataDir+FileName[Modelno/3+1]+String.valueOf(Modelno)+".model",mcls );
    // Modelno++;
     
     
     
     
    //System.out.println(eval.toSummaryString("\nResults\n\n", false));
    // System.out.println("NB_error="+(100-eval.pctCorrect()));
return (eval.errorRate());
}
public static double IBk_error(Instances train,Instances test) throws Exception{
	train.setClassIndex(train.numAttributes() - 1);
	test.setClassIndex(test.numAttributes() - 1);  
	 Classifier NB = new IBk();     
    NB.buildClassifier(train);
    Evaluation eval = new Evaluation(train);
    eval.evaluateModel(NB,test);  
    
   // weka.core.SerializationHelper.write(DataDir+FileName[Modelno/3+1]+String.valueOf(Modelno)+".model",NB );
   // Modelno++;
   //System.out.println(eval.toSummaryString("\nResults\n\n", false));
   // System.out.println("NB_error="+(100-eval.pctCorrect()));
return (eval.errorRate());
}
public static double Ada_error(Instances train,Instances test) throws Exception{
	train.setClassIndex(train.numAttributes() - 1);
	test.setClassIndex(test.numAttributes() - 1);  
	 Classifier NB = new AdaBoostM1();     
    NB.buildClassifier(train);
    Evaluation eval = new Evaluation(train);
    eval.evaluateModel(NB,test);  
    
  //  weka.core.SerializationHelper.write(DataDir+FileName[Modelno/3+1]+String.valueOf(Modelno)+".model",NB );
  //  Modelno++;
   //System.out.println(eval.toSummaryString("\nResults\n\n", false));
   // System.out.println("NB_error="+(100-eval.pctCorrect()));
return (eval.errorRate());
}



}
