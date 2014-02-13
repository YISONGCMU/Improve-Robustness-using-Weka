package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

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

public class MileStone4 {
	static String DataDir="/Users/songyi/Desktop/milestone5b/";
	static int Modelno=1;	
	static String []FileName=new String[100];
	static int fN=40;
public static void main(String args[]) throws Exception{
	File f=new File("/Users/songyi/Desktop/milestone5data/") ;
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
	 System.out.println(" train instance num "+train.numInstances()+" test instance num "+test.numInstances()+" attributes num"+train.numAttributes());
	 
	 int seed = 0;          // the seed for randomizing the data
	 int folds = 2;         // the number of folds to generate, >=2 
	 
	 Random rand = new Random(seed);   // create seeded number generator
	 Instances randData = new Instances(train);   // create copy of original data
	 randData.randomize(rand);         // randomize data with number generator
	 train = randData.trainCV(folds, 0);
	 test = randData.testCV(folds, 0);
	 
	 
	 
	value[i/2]=Output(train,test);
	
	}
	
	
	double mean=0;
	double Max=0;
	double sum=0;
	for (int i=1;i<=fN/2;i++){
		System.out.println("No."+i+"Dataset's errorC/errorNB  "+value[i]);
		if (value[i]>Max) Max=value[i];
		sum=sum+value[i];
	}
	mean=sum/(fN/2);
	System.out.println("Mean"+mean);
	System.out.println("Max"+Max);
	
	
}
	
public static double Output(Instances train,Instances test) throws Exception{

	 double A=NB_Error(train,test);
	
	// L3.0 = Ad tree   
	 
	    train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		
		
			 
	     Evaluation eval = new Evaluation(train);
	     
	     MultiClassClassifier mcls = new MultiClassClassifier();
	     
	     Classifier adtree=new ADTree();
	     
	     mcls.setClassifier(adtree);
	     mcls.buildClassifier(train);
	     eval.evaluateModel(mcls, test);
		
		
		double C=eval.errorRate();
	    double CC=C;
	    

	    
	   
	//L4.0 = stacking:
	   	
	   	String optionClassifier=" -B "+"weka.classifiers.meta.MultiClassClassifier -W weka.classifiers.trees.ADTree"
	   	+" -B "+"weka.classifiers.meta.LogitBoost"
	   			//+" -B "+"weka.classifiers.trees.LMT"
	   	             +" -B "+"weka.classifiers.meta.Dagging"
	   	;
	   	
	   	
	   Classifier L4=new Vote();
	   
	   L4.setOptions(Utils.splitOptions(optionClassifier));
	   
	   L4.buildClassifier(train);
	   Evaluation eval1= new Evaluation(train);
	   eval1.evaluateModel(L4, test);
       double L4Error=eval1.errorRate();
       
       System.out.println("L4 Error Rate="+L4Error/A);
 
       
       
	//---------------------------------------------------	 		   	
	     System.out.println("ADtree Error Rate="+CC/A);
	     
	     
//	     weka.core.SerializationHelper.write(DataDir+FileName[Modelno]+"1.model",mcls );
	     
	     
	     
	 //    outPut(mcls,test,FileName[Modelno]+"-LB");
	 //    outPut(L4,test,FileName[Modelno]+"-L5");
	     Modelno++;
	     
	     
	     
	     
	     
	 return CC/A;
	
}

public static double NB_Error(Instances train,Instances test) throws Exception{
	train.setClassIndex(train.numAttributes() - 1);
	test.setClassIndex(test.numAttributes() - 1); 
		
	 Classifier NB = new NaiveBayes();     
     NB.buildClassifier(train);
     Evaluation eval = new Evaluation(train);
     eval.evaluateModel(NB,test);  
      
return (eval.errorRate());
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




