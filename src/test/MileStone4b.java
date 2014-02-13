package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.ADTree;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class MileStone4b {
	static String DataDir="/Users/songyi/Desktop/milestone4b/";
	static int Modelno=1;	
	static String []FileName=new String[100];
	static int fN=24;
public static void main(String args[]) throws Exception{
	File f=new File("/Users/songyi/Desktop/ms4/") ;
	File[] fd=f.listFiles();
	//System.out.println(fd.length);
	double []value=new double[100];
	
	for (int i=2;i<=fN;i=i+2){
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
	for (int i=1;i<=fN/2;i++){
		System.out.println(/*"No."+i+"Dataset's errorC/errorNB"*/value[i]);
		if (value[i]>Max) Max=value[i];
		sum=sum+value[i];
	}
	mean=sum/(fN/2);
	System.out.println("Mean"+mean);
	System.out.println("Max"+Max);
	
	
}
	
public static double Output(Instances train,Instances test) throws Exception{

    //double A=NB_Error(train,test);
	
	// L3.0 = Ad tree   
	 
	    train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		
		
			 
	    // Evaluation eval = new Evaluation(train);
	     
	     MultiClassClassifier mcls = new MultiClassClassifier();
	     
	     Classifier adtree=new ADTree();
	     
	     mcls.setClassifier(adtree);
	     mcls.buildClassifier(train);
	    // eval.evaluateModel(mcls, test);
		
		
		//double C=eval.errorRate();
	    //double CC=C;
	    

	    
	   
	//L4.0 = stacking:
	   	
	   	String optionClassifier=" -B "+"weka.classifiers.meta.MultiClassClassifier -W weka.classifiers.trees.ADTree"
	   	+" -B "+"weka.classifiers.meta.LogitBoost"
	   			+" -B "+"weka.classifiers.trees.LMT"
	   	             +" -B "+"weka.classifiers.meta.Dagging"
	   	;
	   	
	   	
	   Classifier L4=new Vote();
	   
	   L4.setOptions(Utils.splitOptions(optionClassifier));
	   
	   L4.buildClassifier(train);
	  // Evaluation eval1= new Evaluation(train);
	  //  eval1.evaluateModel(L4, test);
       
	   // double L4Error=eval1.errorRate();      
      // System.out.println("L4 Error Rate="+L4Error);
 
       
       
	//---------------------------------------------------	 		   	
	   //   System.out.println("ADtree Error Rate="+CC);
	     
	     
//	     weka.core.SerializationHelper.write(DataDir+FileName[Modelno]+"1.model",mcls );
	     
	     
	     
	    outPut(mcls,test,FileName[Modelno]+"0");
	    outPut(L4,test,FileName[Modelno]+"1");
	     Modelno++;
	     
	     
	     
	     
	     
	 return 0;//L4Error/A;
	
}



public static void outPut(Classifier L4,Instances test,String name) throws Exception{

	
 FileWriter fw= new FileWriter(DataDir+name+".predict",true);
 BufferedWriter bw= new BufferedWriter(fw);
 
 
 for(int i=0;i<test.numInstances();i++){
	 
	 double label=L4.classifyInstance(test.instance(i));
	 bw.write(Double.toString(label));
	 bw.newLine();
	 bw.flush();
     }
 
     bw.close();
     fw.close();
 
 }



}




