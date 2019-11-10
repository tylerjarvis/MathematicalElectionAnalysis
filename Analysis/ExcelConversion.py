import sqlite3
import csv
import uuid
import sys
import os.path
import functools
import re

DEBUG_LOGGING = False

def is_blank_line(row):
    if row == '': return True
    return type(row) == list and all(cell == '' for cell in row)

class PrecinctConversion(object):
    def __init__(self, lines):
        self.lines = lines

        self.Races : List[str] = []
        self.Precinct : str = None
        self.Candidates : list[str] = []
        self.Results : List[tuple(str,str,int)] = [] # (candidate, race, total)

    def Parse1AB(self, lines, type):
        # race should be the first line
        race = lines[0][0]
        self.Races.append(race)

        # candidates/choices should be the fourth through the end
        for line in lines[3:]:
            if type == 'A':
                candidate, blank, votes, *rest = line
            elif type == 'B':
                candidate, votes, *rest = line

            if candidate in ['Summary Results Report', 'Total Votes Cast']:
                # done processing candidates
                break

            try:
                votesInt = int(votes)
            except:
                raise ValueError("Failed to parse vote string (%s) for candidate '%s' in race '%s'" % (votes, candidate, race))

            self.Results.append((candidate, race, votesInt))

            if candidate not in self.Candidates:
                self.Candidates.append(candidate)

    def Parse5(self):
        # precinct name on the sixth row
        try:
            self.Precinct = re.match('Summary for (.+), All Counters, All Races', self.lines[5][0], re.IGNORECASE).group(1)
        except:
            summary_for = " ".join(self.lines[5]).rstrip()
            try:
                self.Precinct = re.match('Summary for (.+), All Counters, All Races', summary_for, re.IGNORECASE).group(1)
            except:
                print("Failed to parse precinct name from line: %s" % summary_for)
                print("Expected contents: 'Summary for <precinct name>, All Counters, All Races'")
                return
        if DEBUG_LOGGING: print(f"Parsing precinct '{self.Precinct}'")

        # remove the heading up to and including the number of registered
        # voters, ballots cast
        # typically, this is 9 rows, but sometimes it's 8
        if self.lines[8][0].startswith("Registered Voters"):
           index = 9
        else:
           index = 8
        subset = self.lines[index:]

        def parse_races(lines):
            # remove empty lines at the beginning and end
            while is_blank_line(lines[0]): lines.pop(0)
            while is_blank_line(lines[-1]): lines.pop()

            if lines[0][0].startswith("Registered Voters") or lines[0][0].startswith("Num. Report Precinct"):
                lines.pop(0)
                
            endings = [0] + [i + 1 for (i,line) in enumerate(lines) if is_blank_line(line)] + [None]

            for start, end in zip(endings, endings[1:]):
                race_lines = lines[start:end]

                race = race_lines[0][0]

                # often, the first cell is blank but the second has the race
                # name
                if race == '': race = race_lines[0][1]

                if race == 'STRAIGHT PARTY':
                    continue
                elif race == '':
                    print(f"Encountered empty race for {self.Precinct}")
                    continue



                race_lines = race_lines[2:] # remove race name, 'Total' line
                race_lines = race_lines[4:] # remove 'Number of Precincts', 'Precincts Reporting', 'Times Counted', 'Total
                                            # Votes'
                race_lines = race_lines[:-1] # remove the end-of-race blank line

                for (candidate, blank, party, total, percent) in race_lines:
                    if candidate in ('WRITE-IN'):
                        continue

                    try:
                        total = int(total)
                        self.Results.append([candidate, race, total])
                    except ValueError:
                        print(f"Couldn't convert vote count '{total}' to an integer")
                        print(f"    Candidate: {candidate}")
                        print(f"    Race: {race}")
                        print(f"    Precinct: {self.Precinct}")
                        #raise

                if DEBUG_LOGGING: print(f"    Parsed race '{race}'")
                
        # parse the left column
        left = [line[:5] for line in subset]
        parse_races(left)

        # then the right
        right = [line[6:11] for line in subset]
        parse_races(right)

    def Parse7(self):
        """
        Find the distance between each candidate
        Figure out when race change and candidate change
        """
        vote_col = 5
        cand_col = 4
        race_col = 3
        self.Precinct = self.lines[0][2]

        # find U.S.  SENATE and first candidate
        for i in range(len(self.lines)):
            if self.lines[i][race_col] in ["RACE STATISTICS","STRAIGHT PARTY"] or self.lines[i][cand_col] in ["Number of Precincts for Race","Number of Precincts Reporting","Registered Voters"]:
                continue
            else:
                initial_race = self.lines[i][race_col]
                initial_can = self.lines[i][cand_col]
                start = i
                break
        #find the distance between "total" lines
        count = -1
        for i in range(start,start + 6):
            if initial_can == self.lines[i][cand_col]:
                count+=1

        race = initial_race
        candidate = initial_can
        i = start
        while i < len(self.lines):
            # find first candidate in race
            if self.lines[i][race_col] in ["RACE STATISTICS","STRAIGHT PARTY"] or self.lines[i][cand_col] in ["Number of Precincts for Race","Number of Precincts Reporting","Registered Voters"]:
                i+=1
            else:
                race = self.lines[i][race_col]
                candidate = self.lines[i][cand_col]
                self.Races.append(race)
                self.Candidates.append(candidate)
                if i + count < len(self.lines):
                    total = int(self.lines[i + count][vote_col])
                    i += (count + 1)
                else:
                    #Only when at the end of the precinct
                    total = int(self.lines[i + count - 1][vote_col])
                self.Results.append([candidate,race,total])



class RaceConversion(object):
    """
    Handles the parsing of CSV copied/exported from Excel and the INSERT into a database
    CSV are assumed to be a single race spanning multiple precincts
    """
    def __init__(self, lines=None):
        self.lines = lines

        self.Race : str = None
        self.Precincts : List[str] = []
        self.Candidates : list[str] = []
        self.Results : List[tuple(str,str,int)] = [] # (candidate, precinct, total)

    def Parse1(self, precinct):
        self.Candidates = [name for (name, count) in self.lines]
        self.Precincts = [precinct]

        for name, count in self.lines:
            self.Results += name, precinct, count
            
    def Parse2(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is empty

        # Get the candidate names.  These are on the same line as 'Registered
        # Voters', etc.
        TotalVotesIdx = 2
        candidate_line = self.lines[2]
        candidates = [ (i, candidate_line[i]) for i in range(TotalVotesIdx + 1, len(candidate_line)) if candidate_line[i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precinct names are listed in rows starting at index 4 and each take
        # up five rows

        # Get the index of the last row of all precincts.  If the last row of
        # our data is the last row of precincts,
        # then the empty string condition will fail, so check for that.
        precinct_end = [i for i in range(4, len(self.lines)) if self.lines[i][0].strip() in ('', 'Total')]
        precinct_end = precinct_end[0] if (len(precinct_end) > 0) else len(self.lines)
        precincts = [(i, self.lines[i][0]) for i in range(4, precinct_end, 5) if self.lines[i][0].strip() != '']
        self.Precincts = [name for (i,name) in precincts]

        for (p,precinct) in precincts:
            # The updated format (as of ~2019-04-13) lacks a 'Totals' column,
            # so this has to be reconstructed from the
            # sum of Normal, Absentee, Early Voting, and Provisional, which are
            # the four rows (in order) after the precinct name.
            # The 'Total Votes' column is +2 from the precinct name.
            for (c, candidate) in candidates:
                counts = [self.lines[p + 1][c], # normal
                    self.lines[p + 2][c], # absentee
                    self.lines[p + 3][c], # early voting
                    self.lines[p + 4][c]  # provisional
                    ]

                # if thehre are no counts reported, skip this
                # precinct+candidate
                if all([count == '' for count in counts]):
                    continue
                
                try:
                    total = sum(map(int, counts))
                except:
                    print("Non-integer vote count value found for race '%s', precinct '%s', and candidate '%s'" % (self.Race, precinct, candidate))
                    print("   Normal: %s" % counts[0])
                    print("   Absentee: %s" % counts[1])
                    print("   Early Voting: %s" % counts[2])
                    print("   Provisional: %s" % counts[3])
                    continue
                self.Results.append((candidate, precinct, total))

    def Parse3(self, precincts):
        self.Race = self.lines[0][0]
        
        self.Precincts = [name for (idx, name) in precincts]
        self.Candidates = self.lines[1]

        for p, precinct in precincts:
            candidate_line = self.lines[1]
            for c in range(len(candidate_line)):
                candidate = candidate_line[c]
                # precinct counts are on the third line
                total = self.lines[p][c]
                self.Results.append((candidate, precinct, int(total)))
        
    def Parse4(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # candidate names are on the third line after 'Precinct', 'Votes Cast'
        candidate_line = self.lines[2]
        candidates = [ (i, candidate_line[i]) for i in range(2, len(candidate_line)) if candidate_line[i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precinct names are listed in rows starting at index 4 and each take
        # up five rows

        # Read from the fourth line to the last precinct; this format includes
        # a Total, which is unnecessary
        precincts = [(i, self.lines[i][0]) for i in range(3, len(self.lines) - 1) if self.lines[i][0].strip() not in ('', 'Totals')]
        self.Precincts = [name for (i,name) in precincts]


        for (p,precinct) in precincts:
            for (c, candidate) in candidates:
                # Non-Salt Lake format 4 includes the votes below the candidate
                total = self.lines[p][c]
                
                try:
                    total = int(total)
                except:
                    print("Non-integer vote count value ('%s') found for race '%s', precinct '%s', and candidate '%s'" % (total, self.Race, precinct, candidate))
                    continue
                self.Results.append((candidate, precinct, total))

    def Parse4SaltLake(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is candidate names
        candidate_line = self.lines[1]
        candidates = [ (i, candidate_line[i]) for i in range(len(candidate_line)) if candidate_line[i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precinct names are listed in rows starting at index 4 and each take
        # up five rows

        # Read from the fourth line to the last precinct; this format includes
        # a Total, which is unnecessary
        precincts = [(i, self.lines[i][0]) for i in range(3, len(self.lines) - 1) if self.lines[i][0].strip() != '']
        self.Precincts = [name for (i,name) in precincts]


        for (p,precinct) in precincts:
            for (c, candidate) in candidates:
                # Total Votes is offset +3 from the candidate's name
                total = self.lines[p][c + 3]
                
                try:
                    total = int(total)
                except:
                    print("Non-integer vote count value ('%s') found for race '%s', precinct '%s', and candidate '%s'" % (total, self.Race, precinct, candidate))
                    continue
                self.Results.append((candidate, precinct, total))

 
class ExcelConversion(object):
    def __init__(self, filename:str, delimiter:str=None):
        """
        @param: filename A CSV/TSV file to read in
        @param: delimiter Optional delimiter to use; if None, it is inferred from the extension
        """
        self.filename = filename
        if delimiter != None:
            self.delimiter = delimiter
        elif filename.endswith(".csv"):
            self.delimiter = ","
        elif filename.endswith(".tsv"):
            self.delimiter = "\t"
        else:
            raise "Delimiter unhandled"

        with open(self.filename, 'r') as fh:
            reader = csv.reader(fh, delimiter= self.delimiter)
            self.lines = [line for line in reader]

        self.Races : list[RaceConversion] = []

    def AddPrecinctConversions(self, p: PrecinctConversion):
        for candidate, race, total in p.Results:
            # find or create a RaceConversion
            matching_race = [rc for rc in self.Races if rc.Race == race]

            if matching_race == []:
                r = RaceConversion()
                r.Race = race
                self.Races.append(r)
            else:
                r = matching_race[0]

            r.Results.append([candidate, p.Precinct, total])

            if candidate not in r.Candidates:
                r.Candidates.append(candidate)

    def Parse1(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """
        def getNameAndCount(cols):
            values = [col for col in cols if col.strip() != '']
            
            if len(values) < 2: return None

            # these values also signal the end of a race
            if values[0] in ['Total Votes Cast', 'Contest Totals']: return None

            if values[0] == 'Voter Turnout':
                pct = float(values[1].replace('%', ''))
                return values[0], pct
            if not values[1].replace(',', '').isnumeric(): return None

            name, count = values[0], int(values[1].replace(',', ''))
            return name, count


        # some Format 1 files have a completely empty A column.  if so, delete
        # it
        if all([line[0].strip() == '' for line in self.lines]):
            self.lines = [line[1:] for line in self.lines]

        # remove all empty lines
        self.lines = [line for line in self.lines if any([col.strip() != '' for col in line])]
               
        subset = self.lines[:]
        precinct = ''

        while len(subset) > 0:
            if subset[0][0] == 'Summary Results Report':
                # skip the next three lines
                subset = subset[3:]

                # and the precinct follows
                precinct, subset = subset[0][0], subset[1:]

            if subset[0][0] == 'STATISTICS':
                subset = subset[5:]

            race = RaceConversion([])

            # read the race name, remove from the set of lines to parse
            race.Race = subset[0][0]
            subset = subset[1:]

            
            # subset[1][0] should be 'Vote for 1' followed by "total, vote %,
            # mail, provisional"
            subset = subset[2:]
                           
            # now candidates
            for i in range(len(subset)):
                line = subset[i]

                nc = getNameAndCount(line)

                if nc is None:
                    subset = subset[i:]
                    break

            # trim any trailing lines
            while len(subset) > 0 and subset[0][0] in ['Total Votes Cast', 'Contest Totals']:
                subset.pop(0)
                
            if race.Race not in ['STATISTICS', 'STRAIGHT PARTY']:
                race.Parse1(precinct)
                self.Races.append(race)




        # format 1 gives data first by precinct then by race, so we're left
        # with a long list of 'Race' objects
        # that are for only a single precinct.  combine these all so we have
        # all precincts together
        race_names = list({r.Race for r in self.Races})
               
        consolidated :List[RaceConversion] = []

        for race_name in race_names:
            races = [r for r in self.Races if r.Race == race_name]

            combined = RaceConversion([])
            combined.Race = race_name

            # simply combine all lists
            for r in races:
                combined.Candidates += r.Candidates
                combined.Precincts += r.Precincts
                combined.Results += r.Results

            # then get distinct values
            combined.Candidates = list(set(combined.Candidates))
            combined.Precincts = list(set(combined.Precincts))
            consolidated.append(combined)

        self.Races = consolidated

    def Parse1AB(self, type):
        """1A and 1B are similar, except with 1B:
        *   "TOTAL" seems guaranteed to be in the second column
        *   Vote counts are in the second column, not third like in 1A
        *   Precinct names are formatted differently (eg, "AL11 Altamont - County", "AL12 Altamont - City"), but this won't be handled differently
        """

        # Some of the 1A files seem to have empty lines.  probably CR/LF issue.
        # start by removing all empty lines
        # This doesn't seem as big a deal with 1B, but it won't hurt.
        self.lines = [line for line in self.lines if any([col.strip() != '' for col in line])]
        
        # Find all indexes of 'TOTAL' and go back two
        # After 'STATISTICS', it's "^,TOTAL,+", but after "Vote For ", it's
        # "^,,TOTAL,+"
        totals = [i - 2 for (i, line) in enumerate(self.lines) if 'TOTAL' in line[1:3]] + [None]
        num_totals = len(totals)

        precinctName, precinct = None, None
        all_precincts = []

        for lineNo, nextTotal in zip(totals, totals[1:]):
            # a 'TOTAL' line can either be preceded by STATISTICS (in which
            # case we want to grab the precinct name from two lines before
            # TOTAL):
            if self.lines[lineNo + 1][0] == 'STATISTICS':
                new_precinct = self.lines[lineNo - 2][0]
                if new_precinct != precinctName:
                    precinctName = new_precinct

                    precinct = PrecinctConversion([])
                    all_precincts.append(precinct)
                    precinct.Precinct = new_precinct

            # or TOTAL is preceded by 'Vote For 1', which is a race
            elif self.lines[lineNo + 1][0] == 'Vote For 1':
                raceLines = self.lines[lineNo:nextTotal]
                try:
                    precinct.Parse1AB(raceLines, type)

                except ValueError as e:
                    print("Encountered error around line %d" % lineNo)
                    print(e)

            else:
                raise ValueError("Unhandled value precedes TOTAL on line %d" % lineNo)

        for precinct in all_precincts:
            self.AddPrecinctConversions(precinct)


    def Parse2(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """
        # Find the line with the race description
        # This 'cleaned' file puts the string "STATISTICS" into the second
        # column, whereas a previous version put "TURN OUT" into the first
        # https://www.dropbox.com/ow/msft/edit/personal/Utah/UtahData/Election_Results/Precinct%20level/2018%20GE%20SOVC/Cleaned%20Data/Adrienne%20%26%20Annika%20conversions/2018%20General%20Election%20-%20Beaver%20Precinct-Level%20Results(1).xlsx?hpt_click_ts=1555170822997
        index = [i for i in range(len(self.lines)) if self.lines[i][1] == 'STATISTICS'][0]
        subset = self.lines[index:]

        # get the names and indexes of each race
        races = [(i, self.lines[index][i]) for i in range(len(self.lines[index])) if self.lines[index][i].strip() != '']

        # dummy record
        races.append((None, None))

        for i in range(len(races) - 1):
            (current_index, current_name) = races[i]
            (next_index, next_name) = races[i + 1]

            if current_name in ('STATISTICS', 'STRAIGHT PARTY'):
                continue

            # slice the list[][] and create a RaceConversion
            raceLines = [line[current_index:next_index] for line in subset]
            race = RaceConversion(raceLines)
            race.Parse2()

            self.Races.append(race)

    def Parse3(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """

        # get the races, skipping the first two (statistics, straight party)
        race_line = self.lines[0]
        races = [(i, race_line[i]) for i in range(len(race_line)) if race_line[i].strip() != ''][2:]

        # dummy record
        races.append([None, None])

        precincts = [(i, self.lines[i][2]) for i in range(2, len(self.lines)) if self.lines[i][2] != 'COUNTY TOTALS']

        for i in range(len(races) - 1):
            r, race = races[i]
            r2, race2 = races[i + 1]

            # get the subset of text just for this race
            subset = [line[r:r2] for line in self.lines if line[2] != 'COUNTY TOTALS']

            r = RaceConversion(subset)
            r.Parse3(precincts)
            self.Races.append(r)
            
    def Parse4SaltLake(self):
        """Format 4 is one race per sheet, so the ExcelConversion simply hands off parsing to Race
        """
        race = RaceConversion(self.lines)
        race.Parse4SaltLake()
        self.Races = [race]
        
    def Parse4(self):
        """Format 4 is one race per sheet, so the ExcelConversion simply hands off parsing to Race
        """
        race = RaceConversion(self.lines)
        race.Parse4()
        self.Races = [race]

    # https://www.dropbox.com/home/Utah/UtahData/Election_Results/Precinct%20level/2008%20GE%20SOVC/Finished%20Conversions/Format%205
    def Parse5(self):
        # Format 5 has two logical columns, each spanning five physical columns
        # with one separating them.
        # Row-wise, a file has multiple precincts and therefore has to first be
        # split up by row, then columns can be concatenated
        page_breaks = [i for i in range(len(self.lines)) if self.lines[i][0] == 'Election Summary Report'] + [None]

        for i in range(len(page_breaks) - 1):
            start, end = page_breaks[i:i + 2]
            subset = self.lines[start:end]

            precinct = PrecinctConversion(subset)
            precinct.Parse5()

            self.AddPrecinctConversions(precinct)

    def Parse6(self):
        pass


    def Parse7(self):
        """
        Splits when precinct changes and sends to precinct conversion
        """
        all_precincts = []
        vote_col = self.lines[0].index("VOTES")
        prec_col = self.lines[0].index("PRECINCT NAME")
        race_col = self.lines[0].index("RACE")

        i = 1
        while i + 1 < len(self.lines):
            precinct_name = self.lines[i][prec_col]
            all_precincts.append(precinct_name)

            start = i
            while self.lines[i][prec_col] == precinct_name and i + 1 < len(self.lines):
                i+=1


            #Now we chop at end of precinct and send to precinct conversion

            if i+1 < len(self.lines):
                precinctlines = self.lines[start:i]
            else:
                #how we get the last row when we reach the end of the file
                precinctlines = self.lines[start:i+1]

            precinct = PrecinctConversion(precinctlines)
            precinct.Parse7()
            self.AddPrecinctConversions(precinct)




    def Dump(self):
        print("Races and candidates:")
        for race in self.Races:
            print(" - " + race.Race)
            for candidate in race.Candidates:
                print("    - " + candidate)
        print()

        r = self.Races[0]
        print("Precincts (from the first race, '%s'):" % r.Race)
        for precinct in r.Precincts:
            print("    " + precinct)
        print()

class ElectionDatabase(object):
    """
    This class is meant to be an interface to the SQLite database created by the T4DataEntry[1]-created database
    for election data[2].
    
    [1] https://github.com/aeshirey/T4DataEntry
    [2] https://github.com/tylerjarvis/MathematicalElectionAnalysis/tree/master/data%20entry
    """
    def __init__(self, filename):
        self.db = sqlite3.connect(filename)

    def GetCandidateId(self, name): return self.__getId('Candidate', 'Name', name)
    def GetDistrictId(self, desc): return self.__getId('District', 'Name', desc)
    def GetElectionId(self, desc): return self.__getId('Election', 'Description', desc)
    def GetPrecinctnId(self, name): return self.__getId('Precinct', 'Name', name)
    def GetRaceId(self, desc): return self.__getId('Race', 'Description', desc)

    def __getId(self, table, column, value):
        """
        Internal method for INSERT/SELECTing values
        """
        c = self.db.cursor()
        q = 'SELECT %sId FROM %s WHERE %s = ?' % (table, table, column)
        c.execute(q, (value,))
        result = c.fetchone()

        if type(result) == tuple:
            return result[0]
        
        # need to insert
        id = str(uuid.uuid4())
        q = 'INSERT INTO %s (%sId,%s) VALUES (?,?)' % (table, table, column)
        c.execute(q, (id, value))
        self.db.commit()
        return id

    def DumpDB(self):
        tables = ('Candidate', 'District', 'Election', 'Precinct', 'Race')
        c = self.db.cursor()
        for table in tables:
            print(table.upper() + ":")

            # get table schema
            c.execute("pragma table_info('%s')" % table)
            col_names = [name for (columnid, name, sqltype, notnull, defaultvalue, pk) in c.fetchall()]
            print(' | '.join(col_names))

            c.execute("SELECT * FROM %s" % table)
            for row in c.fetchall():
                print(row)
            print("")

        print("Database dump complete")

    def InsertResults(self, candidateId, precinctId, raceId, votes):
        """
        Insert a row into the 'Results' table (which tracks the (Candidate, Precinct, Race) and votes for that tuple)
        """
        c = self.db.cursor()
        # Check if we have results for this 3-tuple
        c.execute("SELECT * FROM Results WHERE CandidateId = ? AND PrecinctId = ? AND RaceId = ?", (candidateId, precinctId, raceId))
        result = c.fetchone()
        if type(result) == tuple: return

        q = 'INSERT INTO Results (CandidateId, PrecinctId, RaceId, VotesReceived) VALUES (?, ?, ?, ?)'
        c.execute(q, (candidateId, precinctId, raceId, votes))

    def CommitParsedResults(self, ec: ExcelConversion, election_name: str):
        electionId = self.GetElectionId(election_name)
        for rc in ec.Races:
            raceId = self.GetRaceId(rc.Race)
            for (candidate, precinct, votes) in rc.Results:
                # make sure candidate and precinct records exist
                candidateId = self.GetCandidateId(candidate)
                precinctId = self.GetPrecinctnId(precinct)

                self.InsertResults(candidateId, precinctId, raceId, votes)

            # commit after each race
            self.db.commit()

def Usage(arg0):
    print("Usage: %s <filename.csv> <format> [-sql]")
    print("Format must be one of: 1, 1A, 1B, 2, 3, 4")
    print("If '-sql' is passed, INSERT statements will be output. Otherwise, debugging output.")

if __name__ == '__main__':

    #sys.argv = ['', r'C:\Users\adams\Downloads\Format 5 - Box Elder.2008.SOVC
    #(2).tsv', '5']
    #sys.argv += [r'2018 General Election - Wasatch Precinct-Level
    #Results.csv', '1']

    if len(sys.argv) <= 2:
       Usage(sys.argv[0])
       exit()

    DATABASE = 'election.sqlite'
    ELECTION_NAME = 'General Election'

    sql = any([arg == '-sql' for arg in sys.argv])
    args = [arg for arg in sys.argv if arg != '-sql']

    if sql and not os.path.exists(DATABASE):
        print("Database '%s' was expected but not found. This is necessary for inserting records" % DATABASE)
        exit()

    filename, format = args[1:3]
    
    if not os.path.exists(filename):
        print("File '%s' doesn't exist" % filename)
        Usage(args[0])
        exit()

    if format.upper() not in ['1', '1A', '1B', '2', '3', '4', '5', '6', '7']:
        print("Format '%s' is invalid" % format)
        Usage(args[0])
        exit()

    converter = ExcelConversion(filename)

    try:
        if format == '1': converter.Parse1()
        elif format.upper() == '1A': converter.Parse1AB(type='A')
        elif format.upper() == '1B': converter.Parse1AB(type='B')
        elif format == '2': converter.Parse2()
        elif format == '3': converter.Parse3()
        # format 4 is different for Salt Lake county
        elif format == '4' and 'Salt Lake' in filename: converter.Parse4SaltLake()
        elif format == '4': converter.Parse4()
        elif format == '5': converter.Parse5()
        elif format == '7': converter.Parse7()

        if sql:
            ed = ElectionDatabase(DATABASE)
            ed.DumpDB()
            ed.CommitParsedResults(converter, ELECTION_NAME)
            ed.DumpDB()
        else:
            print("Parsing completed")
            converter.Dump()

    except BaseException as e:
        print("Exception: %s" % e)
    except:
        print("Parsing halted")