(async () => {
	const del = await import('del');

	const deletedFilePaths = await del.deleteAsync(['./report/*.zip']);
	const deletedDirectoryPaths = await del.deleteAsync([
		'./report/__MACOSX',
		'./report/__pycache__',
		'./report/cache',
		'./report/data',
		'./report/model',
	]);

	console.log('Deleted files:\n', deletedFilePaths.join('\n'));
	console.log('\n\n');
	console.log('Deleted directories:\n', deletedDirectoryPaths.join('\n'));
})();
