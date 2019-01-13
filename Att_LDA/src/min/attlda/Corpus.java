package min.attlda;

import java.awt.print.Printable;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import min.util.URLs;


/**
 * AttLDA Corpus 
 * @author mshi2018
 *
 */
public class Corpus
{
	List<int[]> documentList;
	List<double[]> weightList;
	Map<Integer, List<int[]>> doc2senetnceLists;
	Vocabulary vocabulary; 
	Map<Integer, Integer> serviceDocLocalId2CorpudId; //mapping the doc local ids to the corpus ids
	
	public Corpus()
	{
		documentList = new ArrayList<int[]>();
		weightList = new ArrayList<double[]>();
		doc2senetnceLists = new HashMap<Integer, List<int[]>>();
		vocabulary = new Vocabulary();
		serviceDocLocalId2CorpudId = new HashMap<Integer, Integer>();
	}
	
	// Load the corpus
	public void load(String corpusFile)
	{
		try
		{
			File sourceFile = new File(corpusFile);
			BufferedReader br = new BufferedReader(new FileReader(sourceFile));
			if(!sourceFile.getName().equals("APIDescsWeights.txt"))
			{
				System.out.println("load fails! The source file is not matched.");
			}
			else
			{
				String originaLine = "";
				List<String> wordList = null;
				double[] wList = null;
				String [] params = null;
				while((originaLine = br.readLine()) != null)
				{
					params = originaLine.split(" ");
					wordList = new ArrayList<String>();
					wList = new double[params.length];
					for(String word : params)
					{
						wList[wordList.size()] = Double.parseDouble(word.split("=")[1]);
						wordList.add(word.split("=")[0]);
					}
					addDocument(wordList, wList);
				}
				br.close();
			}
		} catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	// return all documents in the corpus
	public int[][] getDocuments()
	{
		return toArray();
	}
	// return weights of words in all documents
	public double[][] getWeights()
	{
		double[][] wights = new double[this.weightList.size()][];
    	for(int i = 0; i < weightList.size(); i++)
    	{
    		wights[i] = weightList.get(i);
    	}
    	return wights;
	}
	// add the description
    public Map<String, Integer> addDocument(List<String> document, double[] wList)
    {
    	Map<String, Integer> wordIds = new HashMap<String, Integer>();
        int[] doc = new int[document.size()];
        int i = 0;
        for (String word : document)
        {
            doc[i++] = vocabulary.getId(word, true);
            wordIds.put(word, doc[i-1]);
        }
        documentList.add(doc);
        weightList.add(wList);
        return wordIds;
    }
    
    // convert documentList to array
    public int[][] toArray()
    {
    	int[][] docs = new int[this.documentList.size()][];
    	for(int i = 0; i < documentList.size(); i++)
    	{
    		docs[i] = documentList.get(i);
    	}
    	return docs;
    }

	// save the vocabulary, document and sentences ids
    public void saveFiles()
    {
    	try
		{
    		// save the vocabulary
			BufferedWriter bw = new BufferedWriter(new FileWriter(URLs.vocabulary));
			bw.write(this.vocabulary.toString());
			bw.flush();
			bw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    // return the vocabulary
    public Vocabulary getVocabulary()
    {
    	return vocabulary;
    }
    
    // return the vocabulary size
    public int getVocabularySize()
    {
    	return vocabulary.size();
    }
    
	public static void main(String[] args) throws Exception
	{
		Corpus corpus = new Corpus();
		corpus.load(URLs.apisFileToken);
		corpus.saveFiles();
	}
}
